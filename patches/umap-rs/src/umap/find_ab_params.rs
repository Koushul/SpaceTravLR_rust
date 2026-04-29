use ndarray::Array1;

/*
  Fit a, b params for the differentiable curve used in lower
  dimensional fuzzy simplicial complex construction. We want the
  smooth curve (from a pre-defined family with simple gradient) that
  best matches an offset exponential decay.

  Mirrors umap-learn's find_ab_params (scipy.optimize.curve_fit
  equivalent via gradient descent).
*/
pub fn find_ab_params(spread: f32, min_dist: f32) -> (f32, f32) {
  let n_points = 300;
  let mut xv = Array1::<f32>::zeros(n_points);
  for i in 0..n_points {
    xv[i] = (spread * 3.0) * (i as f32) / (n_points as f32 - 1.0);
  }

  let mut yv = Array1::<f32>::zeros(n_points);
  for i in 0..n_points {
    let x = xv[i];
    if x < min_dist {
      yv[i] = 1.0;
    } else {
      yv[i] = f32::exp(-(x - min_dist) / spread);
    }
  }

  // Gradient descent for at least 100 000 steps so that slow-moving
  // coordinates (especially b) have time to reach the global minimum.
  // Each iteration is O(n_points) ≈ O(300) — well under 1 ms total.
  let mut a: f64 = 1.5;
  let mut b: f64 = 0.9;
  let learning_rate: f64 = 0.01;

  // Promote xv/yv to f64 for numerical stability during optimisation.
  let xv64: Vec<f64> = xv.iter().map(|&v| v as f64).collect();
  let yv64: Vec<f64> = yv.iter().map(|&v| v as f64).collect();

  let mut prev_a = a;
  let mut prev_b = b;

  for iter in 0..200_000usize {
    let mut grad_a = 0.0f64;
    let mut grad_b = 0.0f64;

    for i in 0..n_points {
      let x = xv64[i];
      let y_true = yv64[i];
      let x_2b = x.powf(2.0 * b);
      let denom = 1.0 + a * x_2b;
      let y_pred = 1.0 / denom;
      let error = y_pred - y_true;
      grad_a += 2.0 * error * (-x_2b / (denom * denom));
      if x > 0.0 {
        let ln_x = x.ln();
        grad_b += 2.0 * error * (-2.0 * a * x_2b * ln_x / (denom * denom));
      }
    }

    let new_a = (a - learning_rate * grad_a / n_points as f64).clamp(1e-4, 20.0);
    let new_b = (b - learning_rate * grad_b / n_points as f64).clamp(1e-4, 20.0);

    let da = (new_a - prev_a).abs();
    let db = (new_b - prev_b).abs();
    prev_a = a;
    prev_b = b;
    a = new_a;
    b = new_b;

    // Converged when both parameters stop moving.
    if iter > 1000 && da < 1e-9 && db < 1e-9 {
      break;
    }
  }

  (a as f32, b as f32)
}
