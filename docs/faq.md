# Frequently Asked Questions


??? question "What makes SpaceTravLR different?"
    SpaceTravLR predicts **cell-extrinsic** effects: how perturbing a gene in one cell type changes the transcriptome of neighboring, unperturbed cells. Most in-silico perturbation methods only model changes inside the perturbed cell.

??? question "What spatial transcriptomics platforms are supported?"
    We have extensively tested SpaceTravLR using [Slide-seqV2](https://www.nature.com/articles/s41587-020-0739-1), [Slide-tags](https://www.nature.com/articles/s41586-023-06837-4), [VisiumHD](https://www.nature.com/articles/s41588-025-02193-3), [XYZeqV2](https://surveygenomics.com/), [Xenium](https://www.nature.com/articles/s41467-023-43458-x) and [Atera](https://www.10xgenomics.com/platforms/atera).


??? question "What modalities are supported?"
    Currently, the model supports spatial scRNAseq by default. Paired spatial scATACseq can optionally be used to [generate](https://morris-lab.github.io/CellOracle.documentation/tutorials/atac.html) the base regulatory network. Spatial proteomics and other multiomics aren't officially supported yet.

??? question "Can I train a model on just scRNAseq, without spatial data?"
    Yes — use **FreeTravLR**, SpaceTravLR's mean-field implementation. Instead of Gaussian ligand reception from neighboring cells, it models each ligand–receptor pair as **global mean ligand × local receptor**. 

    ```toml
    [ligand_field]
    mode = "meanfield"
    ```

    FreeTravLR recovers broad, population-level ligand–receptor coupling and TF→target structure.


??? question "How many cells can I use for training?"
    SpaceTravLR elegantly scales to very large datasets and has been tested on upto 1 million cells.



??? question "Can I pool multiple samples for training?"
    Yes - concatenate slides into one `.h5ad` with an `obs` column naming each sample, then enable **pool-lasso** training. SpaceTravLR fits one jointly scaled sparse group lasso across all samples, then trains a separate CNN per sample. Outputs land under `conditions/<sample>/` (or `conditions/<condition>/samples/<sample>/` if you also split by `--condition`).

    ```python
    import scanpy as sc

    adatas = {
        "slide1": sc.read_h5ad("slide1.h5ad"),
        "slide2": sc.read_h5ad("slide2.h5ad"),
    }
    adata = sc.concat(adatas, label="sample", keys=adatas.keys())
    adata.write_h5ad("pooled.h5ad")
    ```

    ```bash
    spacetravlr \
      --h5ad pooled.h5ad \
      --output-dir /path/to/output \
      --sample sample \
      --pool-lasso
    ```

    Or in `spaceship_config.toml`:

    ```toml
    [data]
    sample = "sample"

    [training]
    pool_lasso = true
    ```

    `--sample` and `--pool-lasso` must be set together.

??? question "Do I need a GPU?"
    The `spacetravlr` binary will run on any CPU but training the CNN would be very slow. We thus highly recommend using a GPU. SpaceTravLR will automatically detect, configure and use available GPUs. TPUs aren't officially supported yet. Since we use WebGPU, training is driver agnostic.

??? question "Do I need to impute my data?"
    No, but we highly recommend at least smoothing your data for generating the quiver plots. By default, SpaceTravLR uses a custom Rust implementation of [MAGIC](https://github.com/krishnaswamylab/MAGIC), but feel free to try different methods.

??? question "Can I perturb genes beyond transcription factors, ligands or receptors?"
    Yes. `spacetravlr-perturb` can simulate knockout or overexpression of **any gene**.

    By default, training and genome-wide screens only treat TFs, ligands, and receptors as modulators. To include additional genes in the model — for example kinases, cofactors, or ambient-RNA markers — add them under `[grn]` before training:

    ```toml
    [grn]
    extra_modulators = ["MYC", "STAT1"]
    ```

??? question "Is there a Python version?"
    The orignial Python implementation is available at [https://github.com/jishnu-lab/SpaceTravLR](https://github.com/jishnu-lab/SpaceTravLR). **However, we do not plan on maintaining or updating that repo and thus highly recommend using the Rust implementation instead.**



??? question "I have a question. Who do I contact?"
    **For technical questions, bug reports, or feature requests:**  
    - Koushul: [kor11@pitt.edu](mailto:kor11@pitt.edu)  
    - Ally: [alw399@pitt.edu](mailto:alw399@pitt.edu)  

    **For collaboration inquiries:**  
    - Dr. Jishnu Das: [jishnu@pitt.edu](mailto:jishnu@pitt.edu)

