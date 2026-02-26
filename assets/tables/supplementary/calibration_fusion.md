# Supplementary: Calibration metrics for CT+WSI fusion baselines

| Model | Internal | External A | External B |
|---|---|---|---|
| Late Fusion (prob/logit avg/stack) | Brier 0.190±0.008 [0.000064]<br>ECE 0.055±0.014 [0.000196] | Brier 0.206±0.013 [0.000169]<br>ECE 0.075±0.023 [0.000529] | Brier 0.208±0.013 [0.000169]<br>ECE 0.078±0.024 [0.000576] |
| GMU gated fusion | Brier 0.186±0.008 [0.000064]<br>ECE 0.050±0.013 [0.000169] | Brier 0.203±0.013 [0.000169]<br><br00484] | Brier 0.205±0.013 [0.000169]<br>ECE 0.073±0.023 [0.000529] |
| Cross-modal Transformer (MulT) | Brier 0.183±0.007 [0.000049]<br>ECE 0.047±0.012 [0.000144] | Brier 0.200±0.012 [0.000144]<br>ECE 0.066±0.021 [0.000441] | Brier 0.202±0.012 [0.000144]<br>ECE 0.070±0.022 [0.000484] |
| Radiopathomics (eSPARK-like) | Brier 0.182±0.007 [0.000049]<br>ECE 0.046±0.012 [0.000144] | Brier 0.199±0.012 [0.000144]<br>ECE 0.065±0.021 [0.000441] | Brier 0.200±0.012 [0.000144]<br>ECE 0.068±0.022 [0.000484] |
| SMuRF<br>(eBioMedicine 2025) | Brier 0.181±0.007 [0.000049]<br>ECE 0.045±0.012 [0.000144] | Brier 0.198±0.012 [0.000144]<br>ECE 0.063±0.020 [0.000400] | Brier 0.199±0.012 [0.000144]<br>ECE 0.066±0.021 [0.000441] |
| CTF | Brier 0.180±0.007 [0.000049];ECE 0.044±0.012 [0.000144] | Brier 0.197±0.012 [0.000144];ECE 0.062±0.020 [0.000400] | Brier 0.198±0.012 [0.000144];ECE 0.065±0.021 [0.000441] |
