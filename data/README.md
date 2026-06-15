---
license: cc-by-nc-nd-4.0
task_categories:
- image-segmentation
tags:
- Ultrasound
- 3D
- Cardiac
- TEE
- Mitral
- Medical
- MICCAI
pretty_name: Segmentation of the Mitral Valve from 3D TEE
size_categories:
- n<1K
extra_gated_fields:
  Institution: text
  Country: country
  Specific date: date_picker
  I want to use this dataset for:
    type: select
    options: 
      - Research
      - Education
      - label: Other
        value: other
  I agree to use this dataset for non-commercial use ONLY: checkbox
---
# Dataset Card for Dataset Name

<!-- Provide a quick summary of the dataset. -->

![MVSegBanner.png](https://cdn-uploads.huggingface.co/production/uploads/64596d71abdbb77c4c6ca732/-oXiNcTyF140aDn2d8LOu.png)

Goal: Automatically segment mitral valve leaflets from single frame 3D trans-esophageal echocardiography volumes. Challenge hosted at MICCAI 2023.


Challenge event: This was hosted as a challenge workshop at MICCAI 2023.

Additional details: https://www.synapse.org/MVSeg2023

Citation: Carnahan, P., Moore, J., Bainbridge, D., Eskandari, M., Chen, E.C.S., Peters, T.M. (2021). DeepMitral: Fully Automatic 3D Echocardiography Segmentation for Patient Specific Mitral Valve Modelling. In: de Bruijne, M., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2021. MICCAI 2021. Lecture Notes in Computer Science(), vol 12905. Springer, Cham. https://doi.org/10.1007/978-3-030-87240-3_44
Pending updated paper detailing final dataset version.
## Dataset Details
Motivation: Mitral valve (MV) disease is a common pathologic problem occurring in approximately 2 % of the general population but climbing to 10 % in those over the age of 75. The preferred intervention for mitral regurgitation is valve repair, due to superior patient outcomes compared to those following valve replacement. Mitral valve interventions are technically challenging due to the functional and anatomical complexity of mitral pathologies. Repair must be tailored to the patient-specific anatomy and pathology, which requires considerable expert training and experience. Automatic segmentation of the mitral valve leaflets from 3D transesophageal echocardiography (TEE) may play an important role in treatment planning, as well as physical and computational modelling of patient-specific valve pathologies and potential repair approaches. This may have important implications in the drive towards personalized care and has the potential to impact clinical outcomes for those undergoing mitral valve interventions. To date, several groups have developed fully automatic segmentation approaches, achieving reported accuracy of 0.59 mm mean absolute surface distance. However, currently no public datasets are available to develop and compare these segmentation approaches using common data, thus it is difficult to draw conclusions based on reported accuracy alone. We aim to release the first public dataset of 3D TEE volumes of the mitral valve with annotations to enable the broader research community to address the mitral valve segmentation problem.
### Dataset Description
<!-- Provide a longer summary of what this dataset is. -->
The data consists of 3D TEE volumes acquired using Philips Epiq cardiac ultrasound system. Imaging was acquired according to clinical standard of care. Mid-esophageal (en-face) views are used to image the mitral valve using TEE probes. Images were then converted from Philips DICOM to cartesian format using Philips QLab. Data was acquired at King's College Hospital, London, UK. All images are acquired by clinical experts in clinical cardiac imaging and supervised by leaders in the field. Data is pre-processed by converting the Philips DICOM files to cartesian format stored in Nifti format, with all meta-data stripped except for image size and spacing. Any further data processing is left up to the teams.
Training and test case characteristics:
Training and test cases both represent a single-frame 3D ultrasound volume at end-diastole including the mitral valve of 1 patient, with corresponding annotation labeling the valve leaflets in the image.
Labels:
0 - Background
1- Posterior leaflet
2 - Anterior leaflet
The total number of cases is 150 divided as follow:
105 Training,
30 Validation
40 Testing
- **Curated by:** Patrick Carnahan (pcarnah@uwo.ca), Apurva Bharucha
- **License:** CC BY NC ND 4.0
### Dataset Sources
<!-- Provide the basic links for the dataset. -->
- **Repository:** https://www.synapse.org/MVSeg2023
## Citation
<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->
**BibTeX:**
@InProceedings{10.1007/978-3-030-87240-3_44,
author="Carnahan, Patrick
and Moore, John
and Bainbridge, Daniel
and Eskandari, Mehdi
and Chen, Elvis C. S.
and Peters, Terry M.",
editor="de Bruijne, Marleen
and Cattin, Philippe C.
and Cotin, St{\'e}phane
and Padoy, Nicolas
and Speidel, Stefanie
and Zheng, Yefeng
and Essert, Caroline",
title="DeepMitral: Fully Automatic 3D Echocardiography Segmentation for Patient Specific Mitral Valve Modelling",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2021",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="459--468",
isbn="978-3-030-87240-3"
}


