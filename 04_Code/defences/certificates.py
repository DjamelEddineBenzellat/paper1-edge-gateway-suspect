"""Signed model, device certificates, certified groups and t = k (v4 §2-§3, PR §3 item 19, PR §5, D1).

Standard signatures: Ed25519 from the pinned `cryptography` package (requirements_G0.txt).
- Enrollment: each device has a key pair; the cloud signs (device id, public key), the device
  certificate, and keeps the registry. Revoked devices get no certificate.
- Every round the cloud signs, for each gateway, (round, model hash, group list): the round
  certificate. P-TG, P-TG-c and P-PE use it without groups (signed model only); P-TA and P-FR
  also certify the groups (PR §5).
- Before a device trains, it checks: the cloud signature, the round, the model hash against the
  model it received, that the group it is asked to join is the certified one, and that every
  member has a valid device certificate (no Sybil devices). If any check fails, the device
  refuses: it sends nothing.
- t = k: a group sum exists only if all k members sent; otherwise the group aborts, with no
  retry (D1). The cloud sees the aborts and tracks the abort rate (PR §3 item 23).
Simulation note: keys are derived from the run seed, so runs are reproducible. A real
deployment generates them randomly on each party. No global random state is used.
"""
import hashlib
import json

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


def _derived_key(seed, role, ident):
    material = hashlib.sha256(f"paper1-sim-key|{int(seed)}|{role}|{int(ident)}".encode()).digest()
    return Ed25519PrivateKey.from_private_bytes(material)


def _public_hex(key):
    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _verify(public_hex, payload, signature_hex):
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_hex)).verify(bytes.fromhex(signature_hex), payload)
        return True
    except (InvalidSignature, ValueError):
        return False


def model_sha256(vec):
    """Hash of the model the cloud sends (float32 bytes)."""
    return hashlib.sha256(np.ascontiguousarray(vec, dtype=np.float32).tobytes()).hexdigest()


def device_public_hex(seed, device):
    """The device's own public key (in a deployment, the device makes its key pair)."""
    return _public_hex(_derived_key(seed, "device", device))


class Cloud:
    """The cloud's signing key and device registry."""

    def __init__(self, seed):
        self._key = _derived_key(seed, "cloud", 0)
        self.public_hex = _public_hex(self._key)
        self.registry = {}

    def _sign(self, obj):
        payload = _canonical(obj)
        return {"payload": payload.decode(), "sig": self._key.sign(payload).hex()}

    def enroll(self, device, public_hex):
        """Device certificate: the cloud signs (device id, public key)."""
        self.registry[int(device)] = public_hex
        return self._sign({"device": int(device), "pub": public_hex})

    def revoke(self, device):
        self.registry.pop(int(device), None)

    def sign_round(self, rnd, model_vec, gateway, groups=None):
        """Round certificate for one gateway: (round, model hash, group list or none)."""
        return self._sign({"round": int(rnd), "gateway": int(gateway), "model_sha256": model_sha256(model_vec),
                           "groups": [[int(d) for d in g] for g in groups] if groups is not None else None})


def device_check(round_cert, cloud_public_hex, rnd, model_vec, gateway, device, group=None, device_certs=None):
    """What one device checks before it trains. Returns (ok, reason).

    group: the members the device is asked to run secure aggregation with (P-TA, P-FR), or None.
    device_certs: {device id: device certificate} of those members.
    """
    if not _verify(cloud_public_hex, round_cert["payload"].encode(), round_cert["sig"]):
        return False, "round certificate: bad cloud signature"
    body = json.loads(round_cert["payload"])
    if body["round"] != int(rnd) or body["gateway"] != int(gateway):
        return False, "round certificate: wrong round or gateway"
    if body["model_sha256"] != model_sha256(model_vec):
        return False, "model differs from the signed model"
    if group is None:
        return True, "ok"
    members = sorted(int(d) for d in group)
    if body["groups"] is None or members not in [sorted(g) for g in body["groups"]] or int(device) not in members:
        return False, "group is not certified"
    for m in members:
        cert = (device_certs or {}).get(m)
        if cert is None or not _verify(cloud_public_hex, cert["payload"].encode(), cert["sig"]):
            return False, f"member {m}: no valid device certificate (Sybil?)"
        if json.loads(cert["payload"])["device"] != m:
            return False, f"member {m}: certificate is for another device"
    return True, "ok"


def complete_groups(groups, sent):
    """t = k: groups whose k members all sent, and the aborted groups (no retry)."""
    sent = set(int(d) for d in sent)
    complete = [g for g in groups if all(int(d) in sent for d in g)]
    aborted = [g for g in groups if not all(int(d) in sent for d in g)]
    return complete, aborted
