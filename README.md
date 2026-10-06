<img src="Doggybone/Resources/Icons/Doggybone_help_logo.png" alt="Doggybone 3D logo" width="200"/>

# Doggybone 3D

**Doggybone 3D** is a 3D Slicer module for the canine hind limb. It contains the canine hind-limb
angle measurements of BoneAngleMeter and extends them with

- corrective osteotomy planning: closing / opening wedge, bisecting principle, contact-congruent
  alignment (hinge closure and derotation about the contact-face normal), in-plane fine shift with
  a gizmo, prediction of the postoperative angles,
- patient-specific cutting-guide design: footprint on the bone surface (suggestion, brush and
  lasso), extrusion into the guide body with fit gap and wall thickness,
- segmentation clean-up in 3D (lasso, removal of small islands).

In Slicer it appears under **Orthopedics > Doggybone 3D**.

> **For research use only.** Doggybone 3D is not a certified medical device. All measurements and
> planning results must be verified by the surgeon.

**Author:** Matthias Frank, Kleintierklinik Frank & Partner, Freiburg, Germany

**Based on** [BoneAngleMeter](https://github.com/jburgp/BoneAngleMeter_public) by Juliette Burg
Personnaz and Andreas Voigt, developed during Juliette Burg Personnaz' doctorate at CTK, LMU Munich.

**License:** MIT (see [LICENSE](LICENSE)) - Copyright (c) 2022 Juliette Burg Personnaz, Andreas
Voigt; Copyright (c) 2026 Matthias Frank, Kleintierklinik Frank & Partner.

## Installation

1. Install [3D Slicer](https://download.slicer.org/).
2. Download this repository (green **Code** button > **Download ZIP**, or the `.zip` of the latest release) and unpack it, e.g. to `Documents`.
3. In Slicer open **Modules > Developer Tools > Extension Wizard** and click **Select Extension**.
4. Choose the unpacked folder and confirm.
5. Doggybone 3D now appears under **Modules > Orthopedics > Doggybone 3D**.
