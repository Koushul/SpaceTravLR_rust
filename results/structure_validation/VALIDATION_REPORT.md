# Tissue-structure received-ligand validation

- Gaussian radius: **200.0**
- Ligands per dataset: **30** (highly expressed / variable genes)

## Method

Spatial ground truth uses SpaceTravLR's received-ligand kernel
`(1/N) Σ_j scale·exp(-d²/2r²)·expr[j,l]`.
Structure inference replaces per-cell Gaussian weights with type-conditional
expectations `Ŝ[receiver,sender]` learned from a spatial reference, then
multiplies by query type-mean ligand expression.

### Error decomposition

| Method | What it tests |
|---|---|
| `type_mean_oracle` | True per-cell structure × type-mean expression (heterogeneity ceiling) |
| `structure_pooled` | Same-sample type-averaged neighborhoods |
| `expression_matched` | Expression-kNN niche matching to transfer per-cell S |
| `abundance_baseline` | Type frequencies only (no spatial architecture) |
| `structure_transfer` | Cross-sample / cross-replicate structure reuse |
| `expression_matched_transfer` | Cross-sample expression-matched niche transfer |
| `query_self_structure` | Upper bound using the query's own spatial structure |

## Results

### Same-sample recovery

                              dataset             method    mae  pearson_mean  rel_mae  slope  soft_cosine  spearman_mean  type_pearson_mean
      SlideSeqV2_mouse_lymphnode.h5ad abundance_baseline 0.0025       -0.0200   0.5211 0.9921       0.9346        -0.0128             0.2571
      SlideSeqV2_mouse_lymphnode.h5ad expression_matched 0.0022        0.3700   0.4626 0.9750       0.9346         0.3562             0.7543
      SlideSeqV2_mouse_lymphnode.h5ad   structure_pooled 0.0024        0.2141   0.4791 1.0033       0.9346         0.2127             0.7700
      SlideSeqV2_mouse_lymphnode.h5ad   type_mean_oracle 0.0015        0.7570   0.3058 1.0182       0.9346         0.7409             0.7700
        Slidetags_human_melanoma.h5ad abundance_baseline 0.4764        0.1233   0.9988 1.0061       0.9819         0.0848             0.8118
        Slidetags_human_melanoma.h5ad expression_matched 0.4772        0.1471   1.0033 0.9925       0.9819         0.1206             0.9263
        Slidetags_human_melanoma.h5ad   structure_pooled 0.4722        0.1814   0.9918 1.0026       0.9819         0.1361             0.9336
        Slidetags_human_melanoma.h5ad   type_mean_oracle 0.1125        0.8871   0.1775 1.0048       0.9819         0.8717             0.9336
          Slidetags_human_tonsil.h5ad abundance_baseline 0.4136        0.2338   1.3015 0.9713       0.9174         0.2465             0.6917
          Slidetags_human_tonsil.h5ad expression_matched 0.3919        0.3609   1.2260 0.9850       0.9174         0.3519             0.9464
          Slidetags_human_tonsil.h5ad   structure_pooled 0.3931        0.3449   1.2298 0.9946       0.9174         0.3384             0.9637
          Slidetags_human_tonsil.h5ad   type_mean_oracle 0.1331        0.8749   0.4724 0.9914       0.9174         0.8922             0.9637
XYZeqV2_mouse_kidney_replicate_1.h5ad abundance_baseline 0.7679        0.2902   1.4013 1.0315       0.9916         0.2767             0.8942
XYZeqV2_mouse_kidney_replicate_1.h5ad expression_matched 0.7265        0.3524   1.4382 1.0180       0.9916         0.3484             0.2337
XYZeqV2_mouse_kidney_replicate_1.h5ad   structure_pooled 0.7542        0.2936   1.3960 1.0123       0.9916         0.2790             0.9351
XYZeqV2_mouse_kidney_replicate_1.h5ad   type_mean_oracle 0.1889        0.9163   0.3041 1.0177       0.9916         0.9184             0.9351
XYZeqV2_mouse_kidney_replicate_2.h5ad abundance_baseline 0.7155        0.1942   1.2608 1.0317       0.9909         0.1944             0.8358
XYZeqV2_mouse_kidney_replicate_2.h5ad expression_matched 0.7039        0.1746   1.2575 1.0062       0.9909         0.1779             0.3375
XYZeqV2_mouse_kidney_replicate_2.h5ad   structure_pooled 0.7104        0.1972   1.2457 1.0185       0.9909         0.1968             0.8671
XYZeqV2_mouse_kidney_replicate_2.h5ad   type_mean_oracle 0.1836        0.8336   0.5110 1.0269       0.9909         0.8501             0.8671
           snrna_germinal_center.h5ad abundance_baseline 0.0879        0.4377   0.3160 1.1801       0.9621         0.4320             0.7140
           snrna_germinal_center.h5ad expression_matched 0.0741        0.5715   0.2758 0.9836       0.9621         0.5614             0.9980
           snrna_germinal_center.h5ad   structure_pooled 0.0776        0.4941   0.2881 1.0209       0.9621         0.4920             0.9934
           snrna_germinal_center.h5ad   type_mean_oracle 0.0114        0.9920   0.0362 1.0208       0.9621         0.9914             0.9934

```
dataset,method,mae,pearson_mean,rel_mae,slope,soft_cosine,spearman_mean,type_pearson_mean
SlideSeqV2_mouse_lymphnode.h5ad,abundance_baseline,0.002500982388781531,-0.019966838832543472,0.5211237046953288,0.9920806237250843,0.9346474844858408,-0.012757383573585187,0.2570846952401935
SlideSeqV2_mouse_lymphnode.h5ad,expression_matched,0.002249196600098247,0.3700388578234177,0.46257964058395656,0.9750211379574404,0.9346474844858408,0.35617782875372406,0.7542574027017263
SlideSeqV2_mouse_lymphnode.h5ad,structure_pooled,0.0023614282926488635,0.21407249641004059,0.47911893701944,1.0033327206060645,0.9346474844858408,0.21270972128045487,0.7699846653509529
SlideSeqV2_mouse_lymphnode.h5ad,type_mean_oracle,0.0014963039396225148,0.7570367775793484,0.305824020269817,1.0182342790243306,0.9346474844858408,0.7409271213488665,0.7699846653509531
Slidetags_human_melanoma.h5ad,abundance_baseline,0.47637639184349817,0.12332428695960794,0.998812549043366,1.0061023596541447,0.9818972277225639,0.08483528654166263,0.8117547145035217
Slidetags_human_melanoma.h5ad,expression_matched,0.47720884191394985,0.14713660402015089,1.0033375364090533,0.9924989553365406,0.9818972277225639,0.12056753488709655,0.9263022390334009
Slidetags_human_melanoma.h5ad,structure_pooled,0.47216666691638504,0.18136388283150756,0.9918070439636647,1.0026028185156,0.9818972277225639,0.13613628991966747,0.9335809843412167
Slidetags_human_melanoma.h5ad,type_mean_oracle,0.11252794606353798,0.8871296082264223,0.17745982091037693,1.004813888826252,0.9818972277225639,0.8717472735906289,0.9335809843412158
Slidetags_human_tonsil.h5ad,abundance_baseline,0.4135733852948745,0.23381889898726285,1.30145400244391,0.9713015529321649,0.9174106701076503,0.24652483991249685,0.6917299277545242
Slidetags_human_tonsil.h5ad,expression_matched,0.391945978134895,0.36087243968449356,1.2259668816868154,0.9849907104996004,0.9174106701076503,0.35194234453558615,0.9463682076131072
Slidetags_human_tonsil.h5ad,structure_pooled,0.3930834430849235,0.3448835741218266,1.229767103923707,0.9946396938211188,0.9174106701076503,0.338377892346996,0.963747652943021
Slidetags_human_tonsil.h5ad,type_mean_oracle,0.13308728984569526,0.8748540251821982,0.47238032397070645,0.9914406581338836,0.9174106701076503,0.8921690330922581,0.9637476529430219
XYZeqV2_mouse_kidney_replicate_1.h5ad,abundance_baseline,0.7679188229224295,0.29022249102565295,1.4013385954293536,1.0314786212950722,0.991619360053343,0.27671260623551835,0.8941962105246908
XYZeqV2_mouse_kidney_replicate_1.h5ad,expression_matched,0.7265463626729429,0.3524335228730155,1.4381914402196654,1.0179567662250482,0.991619360053343,0.3483931024819353,0.23366461406297992
XYZeqV2_mouse_kidney_replicate_1.h5ad,structure_pooled,0.7541579207411184,0.29360001758421667,1.3960499095981105,1.0123059496595548,0.991619360053343,0.27903587072859454,0.935144540636467
XYZeqV2_mouse_kidney_replicate_1.h5ad,type_mean_oracle,0.18892055715936668,0.916254509297744,0.3040752703854864,1.0177398351440754,0.991619360053343,0.9183927385481847,0.9351445406364669
XYZeqV2_mouse_kidney_replicate_2.h5ad,abundance_baseline,0.7154623300208157,0.19417953471849952,1.2608315202153213,1.031737254710083,0.9908996448281449,0.1944333224326659,0.8357988154761391
XYZeqV2_mouse_kidney_replicate_2.h5ad,expression_matched,0.7038991174107824,0.1745502020108252,1.2574684485712075,1.006247359118416,0.9908996448281449,0.17786057381514345,0.3375023148564558
XYZeqV2_mouse_kidney_replicate_2.h5ad,structure_pooled,0.7104377327431948,0.19718164227139484,1.2457114837727252,1.0185456534518995,0.9908996448281449,0.19681689872425695,0.8670784827845288
XYZeqV2_mouse_kidney_replicate_2.h5ad,type_mean_oracle,0.1836268479828806,0.8335575087313445,0.5110009666592583,1.0269233920406904,0.9908996448281449,0.850130944082736,0.8670784827845295
snrna_germinal_center.h5ad,abundance_baseline,0.08790801105883783,0.43772752735568493,0.31602875289109,1.1800898199041165,0.9620633166717064,0.4320309009833441,0.7140100785309776
snrna_germinal_center.h5ad,expression_matched,0.07407157207080942,0.5715363979286001,0.2757793248707725,0.9835855299247638,0.9620633166717064,0.5613709071464236,0.9980096690550366
snrna_germinal_center.h5ad,structure_pooled,0.07756523519797132,0.49408224154424407,0.28811872317169995,1.0208712141697731,0.9620633166717064,0.4920263940698507,0.9933574614794772
snrna_germinal_center.h5ad,type_mean_oracle,0.011406358460397628,0.9920168441108844,0.03618843529250959,1.020808761718887,0.9620633166717064,0.9913687143725932,0.9933574614794799

```

Note: `type_pearson_mean` scores type-averaged received ligands (the natural estimand without coordinates). Cell-level Pearson is limited by irreducible within-type niche variation.

### Cross-sample structure transfer

              experiment                                                           dataset                      method    mae  pearson_mean  rel_mae  slope  soft_cosine  spearman_mean  structure_matrix_cosine  type_pearson_mean
      replicate_transfer XYZeqV2_mouse_kidney_replicate_1→XYZeqV2_mouse_kidney_replicate_2          abundance_baseline 0.7866        0.0838   0.8686 1.4345       0.9950         0.0768                   0.9697             0.2695
      replicate_transfer XYZeqV2_mouse_kidney_replicate_1→XYZeqV2_mouse_kidney_replicate_2 expression_matched_transfer 0.7840        0.0842   0.8719 1.4071       0.9684         0.0729                   0.9697             0.5052
      replicate_transfer XYZeqV2_mouse_kidney_replicate_1→XYZeqV2_mouse_kidney_replicate_2        query_self_structure 0.5923        0.1421   1.2415 1.0174       0.9950         0.1316                   0.9697             0.7815
      replicate_transfer XYZeqV2_mouse_kidney_replicate_1→XYZeqV2_mouse_kidney_replicate_2          structure_transfer 0.7841        0.0849   0.8638 1.4165       0.9684         0.0806                   0.9697             0.3500
      replicate_transfer XYZeqV2_mouse_kidney_replicate_2→XYZeqV2_mouse_kidney_replicate_1          abundance_baseline 1.3000        0.0712   1.0489 0.6439       0.9947         0.0685                   0.9594            -0.6570
      replicate_transfer XYZeqV2_mouse_kidney_replicate_2→XYZeqV2_mouse_kidney_replicate_1 expression_matched_transfer 1.3381        0.1368   1.0325 0.6318       0.9663         0.1351                   0.9594            -0.4797
      replicate_transfer XYZeqV2_mouse_kidney_replicate_2→XYZeqV2_mouse_kidney_replicate_1        query_self_structure 0.4888        0.1639   0.3549 1.0047       0.9947         0.1571                   0.9594             0.9392
      replicate_transfer XYZeqV2_mouse_kidney_replicate_2→XYZeqV2_mouse_kidney_replicate_1          structure_transfer 1.2762        0.0715   1.0146 0.6451       0.9663         0.0844                   0.9594            -0.6308
spatial_holdout_transfer                                   SlideSeqV2_mouse_lymphnode.h5ad          abundance_baseline 0.0053       -0.0641   0.5033 0.9873          NaN        -0.0829                   0.9873            -0.0296
spatial_holdout_transfer                                   SlideSeqV2_mouse_lymphnode.h5ad        query_self_structure 0.0049        0.2175   0.4791 1.0084          NaN         0.2091                   0.9873             0.8098
spatial_holdout_transfer                                   SlideSeqV2_mouse_lymphnode.h5ad          structure_transfer 0.0050        0.1646   0.4778 0.9991          NaN         0.1699                   0.9873             0.3974
spatial_holdout_transfer                                     Slidetags_human_melanoma.h5ad          abundance_baseline 1.0220       -0.0023   1.5374 0.9986          NaN         0.0097                   0.9584             0.8777
spatial_holdout_transfer                                     Slidetags_human_melanoma.h5ad        query_self_structure 0.8855        0.1256   0.9891 1.0060          NaN         0.1047                   0.9584             0.9730
spatial_holdout_transfer                                     Slidetags_human_melanoma.h5ad          structure_transfer 1.0235        0.0569   1.5298 0.9974          NaN         0.0440                   0.9584             0.8152
spatial_holdout_transfer                                       Slidetags_human_tonsil.h5ad          abundance_baseline 0.9155        0.1115   1.5486 0.7399          NaN         0.0724                   0.9605             0.3401
spatial_holdout_transfer                                       Slidetags_human_tonsil.h5ad        query_self_structure 0.6346        0.3161   1.2313 0.9880          NaN         0.3015                   0.9605             0.9396
spatial_holdout_transfer                                       Slidetags_human_tonsil.h5ad          structure_transfer 0.9073        0.2208   1.4528 0.7441          NaN         0.2147                   0.9605             0.5470
spatial_holdout_transfer                             XYZeqV2_mouse_kidney_replicate_1.h5ad          abundance_baseline 1.4753        0.1980   0.6152 0.9977          NaN         0.2505                   0.9725             0.0473
spatial_holdout_transfer                             XYZeqV2_mouse_kidney_replicate_1.h5ad        query_self_structure 1.4257        0.2875   0.5910 1.0065          NaN         0.2840                   0.9725             0.9343
spatial_holdout_transfer                             XYZeqV2_mouse_kidney_replicate_1.h5ad          structure_transfer 1.4637        0.1984   0.5940 0.9768          NaN         0.2248                   0.9725             0.1298
spatial_holdout_transfer                             XYZeqV2_mouse_kidney_replicate_2.h5ad          abundance_baseline 1.4701        0.1063   1.0947 1.0916          NaN         0.1155                   0.9748            -0.2148
spatial_holdout_transfer                             XYZeqV2_mouse_kidney_replicate_2.h5ad        query_self_structure 1.3920        0.1732   0.9802 1.0138          NaN         0.1690                   0.9748             0.8134
spatial_holdout_transfer                             XYZeqV2_mouse_kidney_replicate_2.h5ad          structure_transfer 1.4572        0.1170   1.0485 1.0896          NaN         0.1204                   0.9748            -0.1938
spatial_holdout_transfer                                        snrna_germinal_center.h5ad          abundance_baseline 0.1528        0.1701   0.4194 0.6333          NaN         0.1499                   0.9029             0.3753
spatial_holdout_transfer                                        snrna_germinal_center.h5ad        query_self_structure 0.1201        0.3598   0.3346 1.0294          NaN         0.3600                   0.9029             0.9840
spatial_holdout_transfer                                        snrna_germinal_center.h5ad          structure_transfer 0.1592        0.2207   0.4228 0.5925          NaN         0.1999                   0.9029             0.5391

```
experiment,dataset,method,mae,pearson_mean,rel_mae,slope,soft_cosine,spearman_mean,structure_matrix_cosine,type_pearson_mean
replicate_transfer,XYZeqV2_mouse_kidney_replicate_1→XYZeqV2_mouse_kidney_replicate_2,abundance_baseline,0.7865564315968097,0.08377938190518688,0.8686077245570457,1.4345165413452208,0.9950178587747016,0.07681016022161623,0.9696733978943375,0.2695051350460002
replicate_transfer,XYZeqV2_mouse_kidney_replicate_1→XYZeqV2_mouse_kidney_replicate_2,expression_matched_transfer,0.7839839399898372,0.08424618512019473,0.8718648761817288,1.4070647610028202,0.9683812843971286,0.07285519345424381,0.9696733978943375,0.5052357136114125
replicate_transfer,XYZeqV2_mouse_kidney_replicate_1→XYZeqV2_mouse_kidney_replicate_2,query_self_structure,0.5923241884449911,0.1420963149119205,1.2415381559790029,1.0173817773457168,0.9950178587747016,0.13160398869861875,0.9696733978943375,0.781465047113264
replicate_transfer,XYZeqV2_mouse_kidney_replicate_1→XYZeqV2_mouse_kidney_replicate_2,structure_transfer,0.7841231296378083,0.08494437256649624,0.8637983569157522,1.4165054849926242,0.9683812843971286,0.08062245350411666,0.9696733978943375,0.35000044410643194
replicate_transfer,XYZeqV2_mouse_kidney_replicate_2→XYZeqV2_mouse_kidney_replicate_1,abundance_baseline,1.300007060424853,0.07118880878123866,1.048940293312623,0.6438538915161388,0.9947425769243049,0.06852450479610676,0.9593654005580421,-0.6570081557635147
replicate_transfer,XYZeqV2_mouse_kidney_replicate_2→XYZeqV2_mouse_kidney_replicate_1,expression_matched_transfer,1.3381278354833528,0.13677144970016475,1.0324616777037645,0.6318218310474256,0.9662686679168393,0.13509019105365314,0.9593654005580421,-0.47974759054713856
replicate_transfer,XYZeqV2_mouse_kidney_replicate_2→XYZeqV2_mouse_kidney_replicate_1,query_self_structure,0.48880693130952463,0.1638939020149251,0.35488456794177337,1.0046975410016732,0.9947425769243049,0.15711147790617874,0.9593654005580421,0.9391577265147105
replicate_transfer,XYZeqV2_mouse_kidney_replicate_2→XYZeqV2_mouse_kidney_replicate_1,structure_transfer,1.2762196767409482,0.07148765679870572,1.014553890480834,0.6450651875990248,0.9662686679168393,0.08440815669062571,0.9593654005580421,-0.6308012138164049
spatial_holdout_transfer,SlideSeqV2_mouse_lymphnode.h5ad,abundance_baseline,0.005250296294471268,-0.0640505908831465,0.5033034258596154,0.9873230318896355,,-0.08289842740940165,0.9873361345640636,-0.02961115753950509
spatial_holdout_transfer,SlideSeqV2_mouse_lymphnode.h5ad,query_self_structure,0.00491705959625185,0.21747003613419305,0.47914374255591585,1.0084471859942898,,0.2090605064531493,0.9873361345640636,0.8098358409050775
spatial_holdout_transfer,SlideSeqV2_mouse_lymphnode.h5ad,structure_transfer,0.005000347154289176,0.16458063284901084,0.47784763660618507,0.9990504620218479,,0.16991525225560206,0.9873361345640636,0.3973594056060242
spatial_holdout_transfer,Slidetags_human_melanoma.h5ad,abundance_baseline,1.021957540420418,-0.0022570215786337504,1.5373683595041856,0.9985681325201864,,0.009653236452685572,0.9584018337195322,0.8777357332058163
spatial_holdout_transfer,Slidetags_human_melanoma.h5ad,query_self_structure,0.8854907535410967,0.12561918720265872,0.9890758823427944,1.0060213137123786,,0.10466339613750125,0.9584018337195322,0.9729569930951713
spatial_holdout_transfer,Slidetags_human_melanoma.h5ad,structure_transfer,1.0234566767488205,0.05686070937647833,1.5298268145196425,0.9974129031049572,,0.04397466113059953,0.9584018337195322,0.8151796912206039
spatial_holdout_transfer,Slidetags_human_tonsil.h5ad,abundance_baseline,0.9154862633053285,0.11148803015018437,1.548562825589345,0.7398638879165417,,0.07242719990392972,0.9605425426356875,0.34014892164695926
spatial_holdout_transfer,Slidetags_human_tonsil.h5ad,query_self_structure,0.6346165760624743,0.3160503475294181,1.2313012494177074,0.9880026465482578,,0.3015205389946842,0.9605425426356875,0.9395806137133839
spatial_holdout_transfer,Slidetags_human_tonsil.h5ad,structure_transfer,0.9073041810800083,0.22077267415339244,1.4527502819719174,0.7440703515709088,,0.21470055390489198,0.9605425426356875,0.5469527176416542
spatial_holdout_transfer,XYZeqV2_mouse_kidney_replicate_1.h5ad,abundance_baseline,1.4753188202337217,0.19799941868589063,0.6152280027649901,0.9976516555542257,,0.2504970087937777,0.9724711328540426,0.04734166982885311
spatial_holdout_transfer,XYZeqV2_mouse_kidney_replicate_1.h5ad,query_self_structure,1.4256524099302392,0.2874793937472201,0.5909718367473892,1.0064554472981841,,0.28401241694662116,0.9724711328540426,0.9343376038528824
spatial_holdout_transfer,XYZeqV2_mouse_kidney_replicate_1.h5ad,structure_transfer,1.4637064227979462,0.19836881975735957,0.5939945416614266,0.9767609393399967,,0.22484836374752992,0.9724711328540426,0.12984132284593616
spatial_holdout_transfer,XYZeqV2_mouse_kidney_replicate_2.h5ad,abundance_baseline,1.4701289788981546,0.10626836427884898,1.0947004826909044,1.0915851894361366,,0.1154571736117514,0.9748062468777463,-0.21477617842303162
spatial_holdout_transfer,XYZeqV2_mouse_kidney_replicate_2.h5ad,query_self_structure,1.3919939164497082,0.17322236388802775,0.9802314716511675,1.0137747449865708,,0.16899073374183338,0.9748062468777463,0.8134370502581263
spatial_holdout_transfer,XYZeqV2_mouse_kidney_replicate_2.h5ad,structure_transfer,1.4571779137346936,0.11699129184651498,1.0485465223042798,1.0896180220412066,,0.12043308019066726,0.9748062468777463,-0.19383922260948253
spatial_holdout_transfer,snrna_germinal_center.h5ad,abundance_baseline,0.1528364694472738,0.17013824225112711,0.4193814786390936,0.633345378122231,,0.14991739882742489,0.9028981657118506,0.3753206541222552
spatial_holdout_transfer,snrna_germinal_center.h5ad,query_self_structure,0.12010075091453912,0.359818218915685,0.3345612480062545,1.029434894003658,,0.360007156931805,0.9028981657118506,0.9840104114602399
spatial_holdout_transfer,snrna_germinal_center.h5ad,structure_transfer,0.15916376154745493,0.22074868013219812,0.4228127265215587,0.5925227199583728,,0.19985338159249297,0.9028981657118506,0.5391418148324991

```

## Interpretation notes

- Pearson / Spearman near the `type_mean_oracle` means structure pooling loses little.
- Gains over `abundance_baseline` show that tissue architecture (not just composition) matters.
- Cross-replicate transfer should approach `query_self_structure` when tissues match.
- `soft_cosine` / `hard_cosine` score inferred neighbor-type composition vs spatial truth.
