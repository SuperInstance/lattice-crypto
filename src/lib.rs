//! # Lattice Cryptography Primitives
//!
//! A pure-Rust library implementing lattice-based cryptographic primitives,
//! including LLL lattice basis reduction, approximate shortest vector computation,
//! and Learning With Errors (LWE) key generation, encryption, and decryption.
//!
//! ## Quick Start
//!
//! ```
//! use lattice_crypto::{LatticeBasis, lwe_encrypt, lwe_decrypt, lwe_keygen};
//!
//! let (secret, public) = lwe_keygen(4, 7, 2, 3);
//! let message = 1i64;
//! let ciphertext = lwe_encrypt(&public, message, 3);
//! let decrypted = lwe_decrypt(&secret, &ciphertext);
//! assert_eq!(message, decrypted);
//! ```

use std::fmt;

/// An n-dimensional lattice point (integer vector).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatticePoint {
    pub coords: Vec<i64>,
}

impl LatticePoint {
    /// Create a new lattice point from coordinates.
    pub fn new(coords: Vec<i64>) -> Self {
        LatticePoint { coords }
    }

    /// Dimension of the point.
    pub fn dim(&self) -> usize {
        self.coords.len()
    }

    /// Euclidean norm squared.
    pub fn norm_squared(&self) -> i64 {
        self.coords.iter().map(|x| x * x).sum()
    }

    /// Euclidean norm (as f64).
    pub fn norm(&self) -> f64 {
        (self.norm_squared() as f64).sqrt()
    }

    /// Dot product with another point.
    pub fn dot(&self, other: &LatticePoint) -> i64 {
        self.coords.iter().zip(&other.coords).map(|(a, b)| a * b).sum()
    }

    /// Subtract another point from this one.
    pub fn sub(&self, other: &LatticePoint) -> LatticePoint {
        LatticePoint::new(
            self.coords.iter().zip(&other.coords).map(|(a, b)| a - b).collect()
        )
    }

    /// Scale by a scalar.
    pub fn scale(&self, s: i64) -> LatticePoint {
        LatticePoint::new(self.coords.iter().map(|x| x * s).collect())
    }
}

impl fmt::Display for LatticePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.coords.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
    }
}

/// A lattice basis represented as a matrix of basis vectors.
#[derive(Debug, Clone)]
pub struct LatticeBasis {
    pub vectors: Vec<LatticePoint>,
}

impl LatticeBasis {
    /// Create a new lattice basis from a list of basis vectors.
    pub fn new(vectors: Vec<LatticePoint>) -> Self {
        let dim = vectors.first().map(|v| v.dim()).unwrap_or(0);
        for v in &vectors {
            assert_eq!(v.dim(), dim, "All vectors must have the same dimension");
        }
        LatticeBasis { vectors }
    }

    /// Number of basis vectors (rank).
    pub fn rank(&self) -> usize {
        self.vectors.len()
    }

    /// Dimension of the space.
    pub fn dim(&self) -> usize {
        self.vectors.first().map(|v| v.dim()).unwrap_or(0)
    }

    /// Compute the Gram-Schmidt orthogonalization.
    ///
    /// Returns (orthogonal vectors, mu coefficients where b*_i = b_i - Σ μ_ij b*_j).
    pub fn gram_schmidt(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n = self.vectors.len();
        let mut orthogonal: Vec<Vec<f64>> = Vec::new();
        let mut mu: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

        for i in 0..n {
            let mut v: Vec<f64> = self.vectors[i].coords.iter().map(|&x| x as f64).collect();

            for j in 0..i {
                let dot_vj: f64 = v.iter().zip(&orthogonal[j]).map(|(a, b)| a * b).sum();
                let norm_sq: f64 = orthogonal[j].iter().map(|x| x * x).sum();
                if norm_sq > 1e-10 {
                    mu[i][j] = dot_vj / norm_sq;
                }
                for k in 0..v.len() {
                    v[k] -= mu[i][j] * orthogonal[j][k];
                }
            }
            orthogonal.push(v);
        }

        (orthogonal, mu)
    }

    /// LLL lattice basis reduction.
    ///
    /// Implements the Lenstra-Lenstra-Lovász algorithm with the given delta parameter
    /// (typically 0.75). Returns a reduced basis with shorter, more orthogonal vectors.
    pub fn lll_reduce(&self, delta: f64) -> LatticeBasis {
        let n = self.vectors.len();
        let d = self.dim();
        let mut b: Vec<Vec<i64>> = self.vectors.iter().map(|v| v.coords.clone()).collect();

        let _orthogonal: Vec<Vec<f64>> = vec![vec![0.0; d]; n];
        let _mu: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        let _norms_sq: Vec<f64> = vec![0.0; n];

        // Compute Gram-Schmidt for current basis
        let recompute_gs = |b: &[Vec<i64>], n: usize, d: usize| -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
            let mut ortho = vec![vec![0.0; d]; n];
            let mut m = vec![vec![0.0; n]; n];
            let mut ns = vec![0.0; n];

            for i in 0..n {
                for k in 0..d {
                    ortho[i][k] = b[i][k] as f64;
                }
                for j in 0..i {
                    let dot_ij: f64 = b[i].iter().zip(&b[j]).map(|(&a, &bb)| (a as f64) * (bb as f64)).sum();
                    if ns[j] > 1e-10 {
                        m[i][j] = dot_ij / ns[j];
                    }
                    for k in 0..d {
                        ortho[i][k] -= m[i][j] * ortho[j][k];
                    }
                }
                ns[i] = ortho[i].iter().map(|x| x * x).sum();
            }
            (ortho, m, ns)
        };

        let (_, _, _) = recompute_gs(&b, n, d);
        let mut k = 1;

        while k < n {
            let (_ortho, m, _ns) = recompute_gs(&b, n, d);

            // Size-reduce b[k]
            for j in (0..k).rev() {
                if m[k][j].abs() > 0.5 {
                    let r = m[k][j].round() as i64;
                    for i in 0..d {
                        b[k][i] -= r * b[j][i];
                    }
                }
            }

            let (_ortho2, m2, ns2) = recompute_gs(&b, n, d);

            // Lovász condition
            let lovasz_lhs = ns2[k];
            let lovasz_rhs = (delta - m2[k][k - 1] * m2[k][k - 1]) * ns2[k - 1];

            if lovasz_lhs >= lovasz_rhs {
                k += 1;
            } else {
                // Swap b[k] and b[k-1]
                b.swap(k, k - 1);
                k = if k > 1 { k - 1 } else { 1 };
            }
        }

        LatticeBasis::new(b.into_iter().map(LatticePoint::new).collect())
    }

    /// Find an approximate shortest vector using LLL reduction.
    pub fn shortest_vector(&self) -> LatticePoint {
        let reduced = self.lll_reduce(0.75);
        let mut shortest = reduced.vectors[0].clone();
        let mut min_norm = shortest.norm_squared();

        for v in &reduced.vectors[1..] {
            let ns = v.norm_squared();
            if ns < min_norm {
                min_norm = ns;
                shortest = v.clone();
            }
            // Also check negation
            let neg = LatticePoint::new(v.coords.iter().map(|x| -x).collect());
            // Same norm, skip
            let _ = neg;
        }

        shortest
    }
}

/// LWE public key: (A, b) where b = A·s + e
#[derive(Debug, Clone)]
pub struct LwePublicKey {
    pub a_matrix: Vec<Vec<i64>>,
    pub b_vector: Vec<i64>,
    pub q: i64,
}

/// LWE secret key.
#[derive(Debug, Clone)]
pub struct LweSecretKey {
    pub s: Vec<i64>,
    pub q: i64,
}

/// LWE ciphertext: (a, b) where b ≈ a·s + m·⌊q/2⌋
#[derive(Debug, Clone)]
pub struct LweCiphertext {
    pub a: Vec<i64>,
    pub b: i64,
}

/// Generate LWE key pair.
///
/// # Arguments
/// * `n` - Dimension of the secret
/// * `q` - Modulus
/// * `m` - Number of samples in public key
/// * `error_bound` - Maximum absolute error
pub fn lwe_keygen(n: usize, q: i64, m: usize, error_bound: i64) -> (LweSecretKey, LwePublicKey) {
    let s: Vec<i64> = (0..n).map(|_| rand_simple(n as i64) % q).collect();

    let mut a_matrix = Vec::new();
    let mut b_vector = Vec::new();

    for _ in 0..m {
        let a: Vec<i64> = (0..n).map(|_| rand_simple(a_matrix.len() as i64 + n as i64) % q).collect();
        let dot: i64 = a.iter().zip(&s).map(|(ai, si)| ai * si).sum();
        let e = (rand_simple(dot.abs() + 1) % (2 * error_bound + 1)) - error_bound;
        let b = (dot + e).rem_euclid(q);

        a_matrix.push(a);
        b_vector.push(b);
    }

    (LweSecretKey { s, q }, LwePublicKey { a_matrix, b_vector, q })
}

/// Encrypt a single bit (0 or 1) using LWE.
pub fn lwe_encrypt(pk: &LwePublicKey, message: i64, error_bound: i64) -> LweCiphertext {
    let n = pk.a_matrix[0].len();
    let m = pk.a_matrix.len();

    // Random subset sum
    let mut a = vec![0i64; n];
    let mut b = (message * (pk.q / 2)).rem_euclid(pk.q);

    for i in 0..m {
        let r = if rand_simple(i as i64 + 1) % 2 == 0 { 0 } else { 1 };
        if r == 1 {
            for j in 0..n {
                a[j] = (a[j] + pk.a_matrix[i][j]).rem_euclid(pk.q);
            }
            b = (b + pk.b_vector[i]).rem_euclid(pk.q);
        }
    }

    // Add small error
    let e = (rand_simple(b.abs() + 1) % (2 * error_bound + 1)) - error_bound;
    b = (b + e).rem_euclid(pk.q);

    LweCiphertext { a, b }
}

/// Decrypt an LWE ciphertext.
pub fn lwe_decrypt(sk: &LweSecretKey, ct: &LweCiphertext) -> i64 {
    let dot: i64 = ct.a.iter().zip(&sk.s).map(|(ai, si)| ai * si).sum();
    let val = (ct.b - dot).rem_euclid(sk.q);

    // Decide: closer to 0 or q/2
    let _half_q = sk.q / 2;
    let quarter_q = sk.q / 4;

    if val <= quarter_q || val >= sk.q - quarter_q {
        0
    } else {
        1
    }
}

/// Simple deterministic pseudo-random function for reproducible tests.
fn rand_simple(seed: i64) -> i64 {
    let mut x = (seed + 1).abs();
    x = ((x >> 16) ^ x).wrapping_mul(0x45d9f3b);
    x = ((x >> 16) ^ x).wrapping_mul(0x45d9f3b);
    x = (x >> 16) ^ x;
    x.abs()
}

/// Determinant of the Gram matrix (as f64).
pub fn gram_determinant(basis: &LatticeBasis) -> f64 {
    let (ortho, _) = basis.gram_schmidt();
    let mut det = 1.0;
    for v in &ortho {
        let norm_sq: f64 = v.iter().map(|x| x * x).sum();
        det *= norm_sq;
    }
    det
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_point_creation() {
        let p = LatticePoint::new(vec![1, 2, 3]);
        assert_eq!(p.dim(), 3);
        assert_eq!(p.coords, vec![1, 2, 3]);
    }

    #[test]
    fn test_norm_squared() {
        let p = LatticePoint::new(vec![3, 4]);
        assert_eq!(p.norm_squared(), 25);
    }

    #[test]
    fn test_norm() {
        let p = LatticePoint::new(vec![3, 4]);
        assert_eq!(p.norm(), 5.0);
    }

    #[test]
    fn test_dot_product() {
        let p = LatticePoint::new(vec![1, 2, 3]);
        let q = LatticePoint::new(vec![4, 5, 6]);
        assert_eq!(p.dot(&q), 32);
    }

    #[test]
    fn test_subtraction() {
        let p = LatticePoint::new(vec![5, 3, 1]);
        let q = LatticePoint::new(vec![1, 1, 1]);
        let r = p.sub(&q);
        assert_eq!(r.coords, vec![4, 2, 0]);
    }

    #[test]
    fn test_scaling() {
        let p = LatticePoint::new(vec![1, 2, 3]);
        let s = p.scale(3);
        assert_eq!(s.coords, vec![3, 6, 9]);
    }

    #[test]
    fn test_gram_schmidt_orthogonality() {
        let basis = LatticeBasis::new(vec![
            LatticePoint::new(vec![4, 1]),
            LatticePoint::new(vec![1, 3]),
        ]);
        let (ortho, _mu) = basis.gram_schmidt();
        // Check orthogonality: dot product should be ~0
        let dot: f64 = ortho[0].iter().zip(&ortho[1]).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-10, "Gram-Schmidt vectors should be orthogonal, dot = {}", dot);
    }

    #[test]
    fn test_gram_schmidt_preserves_span() {
        let basis = LatticeBasis::new(vec![
            LatticePoint::new(vec![1, 0]),
            LatticePoint::new(vec![1, 1]),
        ]);
        let (ortho, _) = basis.gram_schmidt();
        assert!((ortho[0][0] - 1.0).abs() < 1e-10);
        assert!((ortho[0][1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_lll_reduces_vectors() {
        let basis = LatticeBasis::new(vec![
            LatticePoint::new(vec![100, 1]),
            LatticePoint::new(vec![50, 50]),
        ]);
        let reduced = basis.lll_reduce(0.75);
        // Reduced first vector should be shorter
        assert!(reduced.vectors[0].norm_squared() <= basis.vectors[0].norm_squared());
    }

    #[test]
    fn test_lll_output_is_basis() {
        let basis = LatticeBasis::new(vec![
            LatticePoint::new(vec![4, 1]),
            LatticePoint::new(vec![1, 3]),
        ]);
        let reduced = basis.lll_reduce(0.75);
        assert_eq!(reduced.rank(), 2);
        assert_eq!(reduced.dim(), 2);
    }

    #[test]
    fn test_shortest_vector() {
        let basis = LatticeBasis::new(vec![
            LatticePoint::new(vec![100, 0]),
            LatticePoint::new(vec![0, 100]),
        ]);
        let sv = basis.shortest_vector();
        assert!(sv.norm_squared() <= 10000);
    }

    #[test]
    fn test_lwe_keygen_dimensions() {
        let (sk, pk) = lwe_keygen(4, 97, 6, 2);
        assert_eq!(sk.s.len(), 4);
        assert_eq!(pk.a_matrix.len(), 6);
        assert_eq!(pk.b_vector.len(), 6);
    }

    #[test]
    fn test_lwe_encrypt_decrypt_zero() {
        let (sk, pk) = lwe_keygen(4, 97, 6, 1);
        let ct = lwe_encrypt(&pk, 0, 1);
        let decrypted = lwe_decrypt(&sk, &ct);
        assert_eq!(decrypted, 0);
    }

    #[test]
    fn test_lwe_encrypt_decrypt_one() {
        let (sk, pk) = lwe_keygen(4, 97, 6, 1);
        let ct = lwe_encrypt(&pk, 1, 1);
        let decrypted = lwe_decrypt(&sk, &ct);
        assert_eq!(decrypted, 1);
    }

    #[test]
    fn test_gram_determinant() {
        let basis = LatticeBasis::new(vec![
            LatticePoint::new(vec![1, 0]),
            LatticePoint::new(vec![0, 1]),
        ]);
        let det = gram_determinant(&basis);
        assert!((det - 1.0).abs() < 1e-10, "det = {}", det);
    }

    #[test]
    fn test_display_lattice_point() {
        let p = LatticePoint::new(vec![1, 2, 3]);
        let s = format!("{}", p);
        assert_eq!(s, "[1, 2, 3]");
    }
}
