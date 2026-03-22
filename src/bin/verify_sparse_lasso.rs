use space_trav_lr_rust::lasso::{GroupLasso, GroupLassoParams};
use ndarray::array;

fn main() -> anyhow::Result<()> {
    // 3 features, 2 are in the same group (0), 1 is in another (1)
    let x = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    // Target is influenced by features 0 and 2, but NOT 1.
    // Features 0 and 1 are in Group 0.
    // If it's Sparse Group Lasso, Feature 0 should stay, Feature 1 should be zeroed out (within Group 0),
    // and Feature 2 should stay.
    let y = array![[1.0], [0.0], [1.0]];
    
    let mut params = GroupLassoParams::default();
    params.groups = vec![0, 0, 1];
    params.l1_reg = 0.01;
    params.group_reg = 0.01;
    params.fit_intercept = false;

    let mut lasso = GroupLasso::new(params);
    lasso.fit(&x, &y, None)?;

    let fitted = lasso.fitted.as_ref().unwrap();
    println!("Coefficients:");
    println!("  Feature 0 (in Group 0): {:.4}", fitted.coef[[0, 0]]);
    println!("  Feature 1 (in Group 0): {:.4}", fitted.coef[[1, 0]]);
    println!("  Feature 2 (in Group 1): {:.4}", fitted.coef[[2, 0]]);

    if fitted.coef[[0, 0]] != 0.0 && fitted.coef[[1, 0]] == 0.0 {
        println!("\n✅ VERIFIED: Sparsity within Group 0 confirmed (Feature 0 non-zero, Feature 1 zeroed out).");
    } else {
        println!("\n❌ FAILED: Sparsity within Group 0 not observed.");
    }

    Ok(())
}
