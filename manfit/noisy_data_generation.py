import numpy as np

def generate_noisy_manifold(manifold_type='sphere', n_samples=1000, noise_level=0.1, 
                           dim=3, radius=1.0, random_state=None):
    """
    Generate clean manifold data with added noise.
    
    Parameters
    ----------
    manifold_type : str
        Type of manifold: 'sphere', 'swiss_roll', 'circle', 's_curve', 'torus'
    n_samples : int
        Number of samples to generate
    noise_level : float
        Standard deviation of Gaussian noise to add
    dim : int
        Ambient dimension (for sphere only, others have fixed dimensions)
    radius : float
        Radius of manifold (where applicable)
    random_state : int or None
        Random seed for reproducibility
        
    Returns
    -------
    clean : ndarray, shape (n_samples, d)
        Clean manifold data
    noisy : ndarray, shape (n_samples, d)
        Noisy manifold data
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if manifold_type == 'sphere':
        clean = generate_sphere(n_samples, dim, radius)
        
    elif manifold_type == 'swiss_roll':
        clean = generate_swiss_roll(n_samples, radius)
        
    elif manifold_type == 'circle':
        clean = generate_circle(n_samples, radius)
        
    elif manifold_type == 's_curve':
        clean = generate_s_curve(n_samples)
        
    elif manifold_type == 'torus':
        clean = generate_torus(n_samples, radius)
        
    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")
    
    # Add Gaussian noise
    noisy = clean + noise_level * np.random.randn(*clean.shape)
    
    return clean, noisy


def generate_sphere(n_samples, dim=3, radius=1.0):
    """Generate uniform points on a sphere"""
    # Generate from normal distribution and normalize
    points = np.random.randn(n_samples, dim)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return radius * points / norms


def generate_circle(n_samples, radius=1.0):
    """Generate points on a 2D circle"""
    theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    # Add small random perturbation to avoid perfect uniformity
    theta = theta + np.random.uniform(-np.pi/n_samples, np.pi/n_samples, n_samples)
    
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y])


def generate_swiss_roll(n_samples, height=1.0):
    """Generate 3D Swiss roll manifold"""
    t = 3 * np.pi / 2 * (1 + 2 * np.random.rand(n_samples))
    h = height * np.random.rand(n_samples)
    
    x = t * np.cos(t)
    y = h
    z = t * np.sin(t)
    
    return np.column_stack([x, y, z])


def generate_s_curve(n_samples):
    """Generate 3D S-curve manifold"""
    t = 3 * np.pi * (np.random.rand(n_samples) - 0.5)
    x = np.sin(t)
    y = 2.0 * np.random.rand(n_samples)
    z = np.sign(t) * (np.cos(t) - 1)
    
    return np.column_stack([x, y, z])


def generate_torus(n_samples, major_radius=1.0, minor_radius=0.3):
    """Generate 3D torus"""
    theta = 2 * np.pi * np.random.rand(n_samples)
    phi = 2 * np.pi * np.random.rand(n_samples)
    
    x = (major_radius + minor_radius * np.cos(theta)) * np.cos(phi)
    y = (major_radius + minor_radius * np.cos(theta)) * np.sin(phi)
    z = minor_radius * np.sin(theta)
    
    return np.column_stack([x, y, z])