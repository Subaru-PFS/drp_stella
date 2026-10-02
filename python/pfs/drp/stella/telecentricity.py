import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

__all__ = ["twoCircleOverlapOnSphere", "twoCircleOverlap", "computeThetaPhiAxes", "getFiberAngle"]


def twoCircleOverlapOnSphere(a1, theta, a2=None):
    """Calculate the area of overlap between two small circles on the unit sphere
    a1, a2: angular radii of two circles (radians)
    theta: angle between two circles' centres (radians)

    returns: desired area

    N.b. very close to the planar two-circle overlap function for a/theta <~ 0.5;
    within 10% or so for a/theta ~ 1

    From Tovchigrechko, A. and Vakser, I.A. 2001. Protein Sci. 10:1572-1583
    """
    if a2 is None:
        a2 = a1

    def csc(t):
        return 1/np.sin(t)

    def cot(t):
        return 1/np.tan(t)

    if a2 > a1:
        a1, a2 = a2, a1

    area = 2*(np.pi -
              np.arccos(np.cos(theta)*csc(a1)*csc(a2) - cot(a1)*cot(a2)) -
              np.arccos(np.cos(a2)*csc(theta)*csc(a1) - cot(theta)*cot(a1))*np.cos(a1) -
              np.arccos(np.cos(a1)*csc(theta)*csc(a2) - cot(theta)*cot(a2))*np.cos(a2)
              )

    return np.where(a1 + a2 < theta, 0.0,
                    np.where(a2 + theta <= a1, 2*np.pi*(1 - np.cos(a2)), area))


def twoCircleOverlap(r1, d12, r2=None):
    """The overlap of two circles, radii r1 and r2, with centres separated by d12
    """
    d12 = np.where(d12 == 0, 1e-10, d12)

    if r2 is None:
        r2 = r1

    if r2 > r1:
        r1, r2 = r2, r1

    x1 = (r1**2 - r2**2 + d12**2)/(2*d12)  # == d12/2 if r1 == r2
    x2 = d12 - x1
    x1 = np.where(x1 < r1, x1, r1)
    x2 = np.where(x2 < r2, x2, r2)

    psi1 = np.arccos(x1/r1)
    psi2 = np.arccos(x2/r2)
    A = r1**2*(psi1 - np.sin(2*psi1)/2) + r2**2*(psi2 - np.sin(2*psi2)/2)

    return np.where(r2 + d12 > r1, A, np.pi*r2**2)


def threeCircleOverlap(r1, r2, r3, d12, d23, psi,
                       plot=False, hsize=2, labels=["R1", "R2", "R3"]):
    """
    Calculate the overlap of three circles of radii r1, r2, r3 (with r1 >= r2 >= r3)
    d12: the distance between r1 and r2's centres
    d23: the distance between r2 and r3's centres
    psi: the angle C1-C2 to C2-C3
    plot: illustrate the geometry
    hsize: if plotting, the width/height of the plot is 2*hsize*r1

    I.e.
    r1 is centred at (0, 0)
    r2 is centred at (d12, 0)
    r3 is centred at (d12 + d23*cos(psi), d23*sin(psi))   (so d23 is the distance from r2's centre to r3's)

    If plotting and d12 is negative note that the geometry is flipped L <--> R (and the axes are flipped too)

    N.b. the algebra is tedious, so I followed https://apps.dtic.mil/sti/pdfs/ADA463920.pdf
    and that's where the notation ("Step 3") comes from.
    """
    assert r1 >= r2 >= r3

    flipped = False
    if d12 < 0:
        flipped = True
        d12 = -d12
        psi = np.pi - psi

    d13 = np.sqrt(d12**2 + d23**2 + 2*d12*d23*np.cos(psi))

    if plot:
        Circle = plt.matplotlib.patches.Circle

        ax = plt.gca()

        if len(labels) != 3:
            raise RuntimeError(f"Please provide three labels; saw {' ,'.join(labels)}")

        zc1, color = (0, 0), "red"
        ax.add_patch(Circle(zc1, r1, color=color, alpha=0.2, label=labels[0]))
        plt.plot([zc1[0]], [zc1[1]], '+', color=color)
        zc2, color = (d12, 0), "green"
        ax.add_patch(Circle(zc2, r2, color=color, alpha=0.2, label=labels[1]))
        plt.plot([zc2[0]], [zc2[1]], '+', color=color)
        zc3, color = (d12 + d23*np.cos(psi), d23*np.sin(psi)), "blue"
        ax.add_patch(Circle(zc3, r3, color=color, alpha=0.2, label=labels[2]))
        plt.plot([zc3[0]], [zc3[1]], '+', color=color)

        if False:
            plt.plot([0, d12, d12 + d23*np.cos(psi), 0], [0, 0, d23*np.sin(psi), 0], color='black')

        plt.legend()

        plt.axhline(0, color='black', alpha=0.1, zorder=-1)
        plt.axvline(0, color='black', alpha=0.1, zorder=-1)

        hsize *= r1
        plt.xlim((hsize, -hsize) if flipped else (-hsize, hsize))
        plt.ylim(-hsize, hsize)
        ax.set_aspect(1)

    # Step 1
    if r1 - r2 >= d12:  # r2 fits inside r1
        return twoCircleOverlap(r2, d23, r3)

    assert (r1 - r2 < d12) & (d12 < r1 + r2)

    # Step 2.  Intersection of r1 and r2
    x12 = (r1**2 - r2**2 + d12**2)/(2*d12)
    y12 = np.sqrt(2*d12**2*(r1**2 + r2**2) - (r1**2 - r2**2)**2 - d12**4)/(2*d12)

    if plot:
        plt.plot([x12], [y12], '.', color='red')
        plt.plot([x12], [-y12], '.', color='red')

    # Step 3
    ctp = (d12**2 + d13**2 - d23**2)/(2*d12*d13)
    ctpp = -(d12**2 + d23**2 - d13**2)/(2*d12*d23)

    if 1 < ctp < 1 + 1e-6:   # numerical problems
        ctp = 1
    if 1 < ctpp < 1 + 1e-6:
        ctpp = 1

    stp = np.sqrt(1 - ctp**2)
    stpp = np.sqrt(1 - ctpp**2)

    # Step 4
    cond4a = (x12 - d13*ctp)**2 + (y12 - d13*stp)**2 < r3**2
    cond4b = (x12 - d13*ctp)**2 + (y12 + d13*stp)**2 > r3**2

    if not (cond4a and cond4b):  # r3 fits in r1 or r2
        if np.cos(psi) > 0:
            return twoCircleOverlap(r1, d13, r3)
        else:
            return twoCircleOverlap(r2, d23, r3)

    # Step 5.  Intersection of r2 and r3
    xp13 = (r1**2 - r3**2 + d13**2)/(2*d13)
    yp13 = -np.sqrt(2*d13**2*(r1**2 + r3**2) - (r1**2 - r3**2)**2 - d13**4)/(2*d13)

    x13 = xp13*ctp - yp13*stp
    y13 = xp13*stp + yp13*ctp

    xpp23 = (r2**2 - r3**2 + d23**2)/(2*d23)
    ypp23 = np.sqrt(2*d23**2*(r2**2 + r3**2) - (r2**2 - r3**2)**2 - d23**4)/(2*d23)

    x23 = xpp23*ctpp - ypp23*stpp + d12
    y23 = xpp23*stpp + ypp23*ctpp

    if plot:
        plt.plot([x13], [y13], '.', color='magenta')
        plt.plot([x23], [y23], '.', color='magenta')

    # Step 6.  Calculate chord lengths
    c1 = np.hypot(x12 - x13, y12 - y13)
    c2 = np.hypot(x12 - x23, y12 - y23)
    c3 = np.hypot(x23 - x13, y23 - y13)
    if plot:
        plt.plot([x12, x13], [y12, y13], color='black', alpha=0.2)
        plt.plot([x12, x23], [y12, y23], color='black', alpha=0.2)
        plt.plot([x23, x13], [y23, y13], color='black', alpha=0.2)

    # Step 7.  Calculate area
    # Is more than half of r3 included in the spherical triangle?
    eq15 = d13*stp - y13 < (y23 - y13)/(x23 - x13)*(d13*ctp - x13)

    # first the included triangle;  corners (x12, y12), (x13, y13), (x23, y23)
    AT = 0.25*np.sqrt((c1 + c2 + c3)*(c2 + c3 - c1)*(c1 + c3 - c2)*(c1 + c2 - c3))
    # Then the circular segments attached to the sides, coming from r1, r2, and r3
    A1 = r1**2*np.arcsin(c1/(2*r1)) - c1/4*np.sqrt(4*r1**2 - c1**2)
    A2 = r2**2*np.arcsin(c2/(2*r2)) - c2/4*np.sqrt(4*r2**2 - c2**2)
    A3 = r3**2*np.arcsin(c3/(2*r3)) - c3/4*np.sqrt(4*r3**2 - c3**2)
    # Allow for the possibility that r3 has more than half of its area included
    if eq15:
        # Note that result in DSTO-TN-0722 is incorrect, as it assumes a particular choice for the arcsin
        A3 = np.pi*r3**2 - A3

    return AT + A1 + A2 + A3


def computeThetaPhiAxes(alpha_theta, beta_theta, alpha_phi, beta_phi):
    """
    Compute the unit vectors for the cobra (i.e. theta axis) and its phi arm

    alpha: rotation in plane of PFI clockwise from x axis
    beta:  rotation out of the plane, angle from z axis

    alpha_theta: float (radian)
       rotation of cobra axis relative to the boresight (z-axis)
    beta_theta: float (radian)
       rotation of cobra axis in plane of PFI clockwise from x axis
    alpha_phi: float (radian)
        rotation of phi arm relative to the boresight (z-axis)
    beta_phi: float (radian)
        rotation of phi arm in plane of PFI clockwise from x axis when theta == 0
    """
    ca, sa, cb, sb = np.cos(alpha_theta), np.sin(alpha_theta), np.cos(beta_theta), np.sin(beta_theta)
    theta_axis = np.array([sa*cb, sa*sb, ca])

    ca, sa = np.cos(alpha_phi), np.sin(alpha_phi)

    boresight = np.array([0, 0, 1])

    cross = [0, 1, 0] if alpha_theta == 0 else np.cross(boresight, theta_axis)
    cross /= np.sqrt(np.dot(cross, cross))
    phi_axis = Rotation.from_rotvec(cross*alpha_phi).apply(theta_axis)
    phi_axis = Rotation.from_rotvec(theta_axis*beta_phi).apply(phi_axis)

    return theta_axis, phi_axis


def getFiberAngle(theta_axis, phi_axis, theta, phi):
    """Return the vector given by rotating the theta and phi motors about {theta,phi}_axis respectively

    One or both theta and phi may be arrays, in which case a list of vectors is returned
    """
    isVector = False
    try:
        theta[0]
        isVector = True
    except (IndexError, TypeError):
        theta = [theta]

    try:
        phi[0]
        isVector = True
    except (IndexError, TypeError):
        phi = [phi]

    if len(theta) != len(phi):
        if len(theta) == 1:
            theta = len(phi)*[theta[0]]
        elif len(phi) == 1:
            phi = len(theta)*[phi[0]]
        else:
            raise RuntimeError(f"Lengths of theta and phi must be 1 or match: {len(theta)=} {len(phi)=}")

    boresight = np.array([0, 0, 1])

    fibers = []
    for t, p in zip(theta, phi):
        # Obey the cobracharmer conventions:
        #  "thetaMove, phiMove: the angle to move away,
        #                       positive/negative values mean moving away from CCW/CW limits"
        t = -t
        p = np.pi - p

        fiber = Rotation.from_rotvec(theta_axis*p).apply(phi_axis)
        fiber = Rotation.from_rotvec(boresight*t).apply(fiber)

        fibers.append(fiber)

    return fibers if isVector else fibers[0]
