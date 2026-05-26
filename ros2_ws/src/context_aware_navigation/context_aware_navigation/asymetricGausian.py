import numpy as np

# Numba is optional — it only accelerates the one-shot proxemic-zone
# computation done at node startup.  Newer numba releases pull in a coverage
# dependency that conflicts with apt's older coverage package, so fall back to
# a no-op decorator if the import is broken on this system.
try:
    from numba import jit  # noqa: F401
except Exception:  # pylint: disable=broad-except
    def jit(func=None, **_kwargs):  # type: ignore[misc]
        if func is None:
            return lambda f: f
        return func


@jit
def initSocialZones(density, sigmaFront, velocities, maxcost, plotsize=3):
    x = np.arange(-plotsize, plotsize, density)  # [m]
    y = np.arange(-plotsize, plotsize, density)
    zones = []
    for vel in velocities:
        zones.append(makeProxemicZone(
            0, 0, x, y, 0, sigmaFront+(1*vel), maxcost))  # TODO only apply velocity if its negative
    return zones


@jit
def asymetricGaus(x=0, y=0, x0=0, y0=0, theta=0, sigmaFront=2, sigmaSide=4/3, sigmaBack=1) -> float:
    angle: float = np.mod(np.arctan2(y-y0, x-x0), 2*np.pi)+theta
    if (abs(angle) >= np.pi/2 and abs(angle) <= np.pi+np.pi/2):
        sigma: float = sigmaBack
    else:
        sigma: float = sigmaFront

    a: float = ((np.cos(theta) ** 2)/(2*sigma ** 2)) + \
        ((np.sin(theta) ** 2)/(2*sigmaSide ** 2))
    b: float = (np.sin(2*theta)/(4*sigma ** 2)) - \
        (np.sin(2*theta)/(4*sigmaSide ** 2))
    c: float = ((np.sin(theta) ** 2)/(2*sigma ** 2)) + \
        ((np.cos(theta) ** 2)/(2*sigmaSide ** 2))

    return np.exp(-(a*(x-x0) ** 2+2*b*(x-x0)*(y-y0)+c*(y-y0) ** 2))


@jit
def makeProxemicZone(x0, y0, x, y, theta, sigmaFront, maxcost) -> np.ndarray:
    social: np.ndarray = np.zeros((len(x), len(y)), dtype=np.uint8)
    for i in range(len(x)):
        for j in range(len(y)):
            social[j, i] = thresholdCost(asymetricGaus(
                x[i], y[j], x0, y0, theta, sigmaFront), maxcost)
    return social


@jit
def thresholdCost(cost: float, maxcost: float) -> float:
    if cost > asymetricGaus(y=0.5):
        return maxcost
    if cost > asymetricGaus(y=1.0):
        return np.floor(maxcost*asymetricGaus(y=1.0))
    if cost > asymetricGaus(y=1.5):
        return cost*maxcost
    return 0
