import unittest
import numpy as np
from argus_identity import IdentityEngine, IdentityConfig

class IdentityEngineTests(unittest.TestCase):
    def setUp(self):
        cfg = IdentityConfig(match_threshold=0.70, ambiguity_margin=0.03,
                             min_box_area=100, min_crop_width=8,
                             min_crop_height=8, max_lost_frames=20)
        self.engine = IdentityEngine(cfg)
        self.frame = np.zeros((240,320,3), dtype=np.uint8)
        self.frame[20:210,30:130] = (30,80,210)

    def test_output_fields(self):
        r = self.engine.update(self.frame,[{"track_id":1,"bbox":[30,20,130,210],"confidence":0.9,"class_id":0}])[0]
        self.assertIn("global_id",r)
        self.assertIn("identity_status",r)
        self.assertGreater(r["global_id"],0)

    def test_same_tracker_id_keeps_identity(self):
        t={"track_id":1,"bbox":[30,20,130,210],"class_id":0}
        a=self.engine.update(self.frame,[t])[0]["global_id"]
        b=self.engine.update(self.frame,[t])[0]["global_id"]
        self.assertEqual(a,b)

    def test_reset(self):
        self.engine.update(self.frame,[{"track_id":1,"bbox":[30,20,130,210]}])
        self.engine.reset()
        self.assertEqual(self.engine.diagnostics()["known_identities"],0)
        self.assertEqual(self.engine.diagnostics()["frame_index"],-1)

    def test_bad_bbox_raises(self):
        with self.assertRaises(ValueError):
            self.engine.update(self.frame,[{"track_id":1,"bbox":[4,4,2,2]}])

    def test_identity_recovery_after_occlusion(self):
        first=self.engine.update(self.frame,[{"track_id":1,"bbox":[30,20,130,210],"confidence":0.9,"class_id":0}])[0]
        for _ in range(5):
            self.engine.update(self.frame,[])
        recovered=self.engine.update(self.frame,[{"track_id":9,"bbox":[30,20,130,210],"confidence":0.9,"class_id":0}])[0]
        self.assertEqual(recovered["global_id"],first["global_id"])
        self.assertEqual(recovered["identity_status"],"recovered")

    def test_tracker_id_reuse_after_gap(self):
        first=self.engine.update(self.frame,[{"track_id":1,"bbox":[30,20,130,210],"class_id":0}])[0]
        self.engine.update(self.frame,[])
        recovered=self.engine.update(self.frame,[{"track_id":1,"bbox":[30,20,130,210],"class_id":0}])[0]
        self.assertEqual(recovered["global_id"],first["global_id"])
        self.assertEqual(recovered["identity_status"],"recovered")

if __name__ == "__main__":
    unittest.main()
