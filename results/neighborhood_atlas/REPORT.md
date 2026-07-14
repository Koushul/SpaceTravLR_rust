# Neighborhood Grammar Atlas — build report

Technologies: **SlideSeqV2**, **VisiumHD**, **SlideTags** only.

## Entries

| dataset_id | tech | organ | n_ref / n_source | #types | radius |
|---|---|---|---|---|---|
| mouse_ln_slideseqv2 | SlideSeqV2 | lymph_node | 12000 / 22227 | 6 | 402.4 |
| human_tonsil_slidetags | SlideTags | tonsil | 5778 / 5778 | 11 | 213.6 |
| human_tonsil_slidetags_fine | SlideTags | tonsil | 5778 / 5778 | 12 | 281.2 |
| human_melanoma_slidetags | SlideTags | tumor_melanoma | 4804 / 4804 | 7 | 195.4 |
| mouse_kidney_visiumhd | VisiumHD | kidney | 10000 / 10000 | 7 | 269.3 |
| mouse_brain_slideseqv2 | SlideSeqV2 | brain | 12000 / 27359 | 4 | 528.7 |
| mouse_hippocampus_slideseqv2 | SlideSeqV2 | hippocampus | 12000 / 41786 | 12 | 555.6 |
| mouse_brain_slideseqv2_regions | SlideSeqV2 | brain | 12000 / 44541 | 8 | 534.1 |

## Label consistency

All raw labels mapped exhaustively into `ontology.json`. Loader
`assert_label_consistency()` passes.

## Structure matrix cosine (shared harmonized types)

| A | B | cosine | n_shared |
|---|---|---|---|
| LN SlideSeqV2 | tonsil SlideTags | 0.970 | 5 |
| tonsil coarse | tonsil fine | 0.997 | 11 |
| tonsil | melanoma | 0.432 | 4 |
| brain coarse | hippocampus | 0.971 | 2 (Endothelial, Oligodendrocyte) |
| brain coarse | brain regions | 0.672 | 4 |

Lymphoid structure transfers well across species/tech. Melanoma vs tonsil is
weaker at subtype resolution (expected tumor niche) but stronger at lineage
projection (~0.91). Neural entries share glia/endothelium structure.

## Disk location

`data/neighborhood_atlas/` — load via `scripts/load_neighborhood_atlas.py`.
