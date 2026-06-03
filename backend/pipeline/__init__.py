"""Processing pipeline modules.

Phase 1: anonymize  — face blurring
Phase 2: hand_pose  — MediaPipe Hands keypoints
Phase 3: segmentation — VLM task segmentation
Phase 4: events     — derived operational metrics
Phase 7: export     — structured export bundle
"""
