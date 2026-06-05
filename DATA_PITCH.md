# Revisent — Egocentric Manipulation Data for Robot Learning

**One line:** We capture **first-person hand-manipulation demonstrations from real industrial worksites** — waste handling, demanufacturing, the dull/dirty/dangerous jobs you're building robots to do — and deliver them as training-ready data you can't collect yourself.

---

## The problem you have
VLA and humanoid policies are bottlenecked on **diverse, real-world manipulation data**. Teleoperation doesn't scale to the long tail; lab and kitchen datasets don't transfer to messy industrial settings; and you **can't send a capture crew into a live waste-transfer station or an e-waste teardown line.** That data is scarce precisely because it's hard to get.

## What we deliver
Workers wear a camera and do their real job. We turn that footage into **synchronized, training-ready manipulation data**:

| Stream | Detail |
|---|---|
| `observation.images.ego` | First-person video (faces auto-blurred, audio removed) |
| **Hand pose** | 21-pt per hand, the measured, manipulation-relevant signal |
| Arm / wrist / torso | Upper-body pose with per-joint confidence |
| `action` | End-effector (wrist) targets for imitation learning |

Delivered in **LeRobot** (loads in one command), **SMPL-X/MANO**, or raw joints+confidence parquet — your pick.

## Why it's credible (measured, not claimed)
- **Hand accuracy: 22.6 mm PA-MPJPE on AssemblyHands (egocentric)**, 15.0 mm on FreiHAND (third-person) — benchmarked against public ground truth, reproducible.
- **Per-joint confidence + provenance** on every joint: you see exactly what's measured (hands) vs inferred, and can filter/weight accordingly.
- **Tuned for the real world** — blur, fast motion, occlusion, unstructured scenes (not lab conditions).
- **Consent + training rights** attached per clip. Legally yours to train on.

## Why us (the moat)
The pipeline is the easy part — **access is the moat.** We have the operator relationships to instrument **real industrial workforces at scale**, in environments you can't reach. The more we collect, the larger the gap between our data and anything you can replicate.

## Try it
A **sample dataset loads in one command** — verified in vanilla LeRobot (v3.0):
```python
pip install lerobot          # lerobot 0.5.1, verified
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="revisent/ego-manipulation", root=".../sample_bundle/lerobot")
ds[0]   # -> observation.state (51), observation.confidence (17), action (6), task, ...
```
Built with LeRobot's own API, so it drops straight into your training pipeline.
Prefer raw? `pandas.read_parquet` on the joints file needs nothing else. Full
schema, kinematic tree, and per-joint provenance ship with the bundle.

## Honest scope
A head/chest camera **measures hands and torso motion and infers the rest** (it can't see the wearer's legs) — every joint is labeled so you're never guessing. We sell the measured signal (hands + manipulation) and are transparent about the inferred parts. The detector is a strong off-the-shelf model; **our value is the data: real worksites, at scale, that nobody else has.**

---
**Next step:** tell us the tasks/objects you care about and we'll deliver a targeted sample. Pilot volumes available now; scale-up on request.
