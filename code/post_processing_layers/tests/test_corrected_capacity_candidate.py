"""Small deterministic checks; no raster access or production writes."""
import unittest
import numpy as np
from code.post_processing_layers.run_corrected_capacity_candidate import capacity, daily_mean


class CorrectedCapacityTests(unittest.TestCase):
    def test_bounds_and_temperature_direction(self):
        t=np.linspace(5,50,201)
        for rh in (35,45,55):
            c=capacity(t,rh)
            self.assertTrue(np.all((c>=.1)&(c<=1)))
            self.assertTrue(np.all(np.diff(c)<0))

    def test_zero_and_signed_changes(self):
        t=np.array([22.,28.,34.])
        c=capacity(t,45)
        np.testing.assert_array_equal(c-c,np.zeros_like(c))
        self.assertTrue(np.all(capacity(t-.3,45)-c>0))
        self.assertTrue(np.all(capacity(t+.3,45)-c<0))
        np.testing.assert_allclose(100*(capacity(t-.3,45)-c),[.02763389,.97482423,3.23543240],atol=1e-7)

    def test_daily_nonlinearity_and_equal_weights(self):
        t=np.array([28.]);offsets=np.array([-6.,6.])
        daily=daily_mean(t,offsets,45)
        np.testing.assert_allclose(daily,(capacity(t-6,45)+capacity(t+6,45))/2,atol=1e-14)
        self.assertGreater(abs(float(daily[0]-capacity(t,45)[0])),.01)
        daily_gain=100*(daily_mean(t-.3,offsets,45)-daily)
        at_mean_gain=100*(capacity(t-.3,45)-capacity(t,45))
        self.assertGreater(float(daily_gain[0]),float(at_mean_gain[0]))

    def test_lookup_bound_for_toy_daily_series(self):
        grid=np.arange(20.,35.001,.001);offsets=np.array([-3.,0.,4.])
        targets=np.array([20.,20.0005,25.5535,34.9995,35.])
        for rh in (35,45,55):
            lut=daily_mean(grid,offsets,rh)
            approximate=np.interp(targets,grid,lut)
            exact=daily_mean(targets,offsets,rh)
            self.assertLess(float(np.max(abs(approximate-exact)))*200,1e-4)


if __name__=='__main__':unittest.main()
