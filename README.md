# GenEEG — Patient-Adaptive EEG Synthesis for Improved Seizure Detection
[geneeg.full_cropped.pdf](https://github.com/user-attachments/files/22451311/geneeg.full_cropped.pdf)

**Short description**  
GenEEG is a patient-adaptive generative framework that produces high-quality synthetic EEG sequences to improve automated seizure detection. It combines a multi-domain Variational Autoencoder (VAE) to learn a robust latent space with a Latent Diffusion Model (LDM) that generates patient- and state-conditioned EEG (Normal / Preictal / Ictal). Targeted augmentation and sequential fine-tuning address class imbalance and inter-patient variability to improve downstream classifiers.

---

## Key contributions
- Two-stage generative pipeline: **VAE → Latent Diffusion** tailored for EEG synthesis.  
- **Patient-adaptive** and **state-conditioned** generation (Normal / Preictal / Ictal).  
- Targeted augmentation for minority classes to improve seizure detection generalization.  
- Rigorous evaluation on unseen patients and statistical validation showing high similarity between synthetic and real EEG distributions.

---

## Summary of results
Tested on the Siena EEG dataset with unseen-patient evaluation:
- **Accuracy:** 0.89  
- **Macro F1-Score:** 0.86  
- **Macro AUC:** 0.95  
~25% relative improvement (~17 percentage points) in Macro F1 vs. traditional oversampling baselines.

---

## Method overview
1. **Preprocessing & feature extraction** — filtering, artifact handling, epoching, and extraction of spectral/temporal features.  
2. **VAE (multi-domain loss)** — learns a physiologically meaningful latent representation using reconstruction + domain-specific losses.  
3. **Latent Diffusion Model** — conditioned on clinical state + patient features to synthesize realistic latent codes; decoded back to EEG.  
4. **Sequential fine-tuning & augmentation** — generate minority-class examples (preictal/ictal) to balance training sets for a downstream CNN–BiLSTM classifier.  
5. **Evaluation** — unseen-patient splits to measure true generalization; classification metrics + statistical tests (e.g., KS test) comparing real vs synthetic distributions.

---

Dependencies (core)

- Python 3.9+

- PyTorch

- NumPy, SciPy, pandas

- MNE or librosa (EEG utilities)

- scikit-learn

- Matplotlib / seaborn

Manuscript in Peer-Review


