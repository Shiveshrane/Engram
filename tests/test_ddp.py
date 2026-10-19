"""Synchronized-DDP tests: 2 processes with the gloo backend on CPU (runs on a laptop).
On Kaggle, NB01 runs the same checks with NCCL on 2 x T4:
    torchrun --standalone --nproc_per_node=2 -m engramdelta.scripts.ddp_check

Checks (engramdelta/scripts/ddp_check.py): gradient equivalence with single-process full-batch training, parameter
sync across ranks, and exact checkpoint-resume.
"""
import os
import socket
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_ddp_two_process_cpu(tmp_path):
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    if sys.platform == "darwin":       # macOS: the default hostname rendezvous can time out; use loopback
        env["GLOO_SOCKET_IFNAME"] = "lo0"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
           "--master_addr=127.0.0.1", f"--master_port={free_port()}",
           "-m", "engramdelta.scripts.ddp_check", "--device", "cpu", "--workdir", str(tmp_path / "ddp")]
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    print(p.stdout[-4000:], p.stderr[-4000:])
    assert p.returncode == 0, "DDP checks failed (see output above)"
    assert "ALL PASS" in p.stdout
