Figure 2. Two variants of raster based conceptual distributed models (of type 2): ( -A ) FW 1 routes -the discharge directly to the outlet using Maxbas function, and ( B ) FW 2 routes the discharge from -one cell to another following the river network using the Muskingum method.

![Image](geoscience-paper_artifacts%5Cimage_000000_0bb3fab8c73dc60d39d1aefd87fcffa8d95aa7ed8f67ac920355a00c50bb4456.png)

## 3.1.3. Model Structure

Hydrological processes within the adopted spatial discretization (subcatchment and cell) are represented using the same lumped HBV model structure [13]. HBV consists of three tanks (soil moisture, upper response, and lower response) without considering the snow subroutine process (as followed in this study). For each spatial discretization ele -ment (cell), infiltration, soil moisture, percolation, and runoff generation processes are cal -culated. The runoff process is represented using a nonlinear function of actual soil mois -ture and precipitation in the upper reservoir; the nonlinear function simulates the surface runoff and interflow. The baseflow is simulated using a linear function of the lower res -ervoir tank. Both reservoirs interact in series by constant capillary rise and percolation. The primary parameters of the HBV are shown in Table 1, with the range of values used in the calibration for each parameter. More details on the HBV model can be found in [13].

Table 1. HBVparameters and the ranges used in the calibration process.

| Parameter Name   | Description                          | Units   | Lower Bound    | Upper Bound       |
|------------------|--------------------------------------|---------|----------------|-------------------|
| RFCF             | Precipitation correction factor      | -       | 0.93           | 1.3               |
| FC               | Maximum soil moisture storage        | mm      | 150            | 500               |
| Beta             | Nonlinear runoff parameter           | -       | 0.01           | 5                 |
| ETF              | Evapotranspiration correction factor | -       | 0.0            | 1.25              |
| LP               | Limit for potential evaporation      | %       | 0.1            | 0.55              |
| CFLUX            | Maximum capillary rate               | mm/h    | 0.05           | 0.55              |
| K                | Upper storage coefficient            | 1/h     | 0.00055        | 0.008             |
| K1               | Lower storage coefficient            | 1/h     | 0.0035         | 0.012             |
| Α                | Nonlinear response parameter         | -       | 0              | 0.3               |
| Perc             | Percolation rate                     | 1/h     | 0.15           | 0.7               |
| Clake            | Lake correction factor               | %       | 0.85           | 1.15              |
| Maxbas           | Transfer function length             | h       | 1              | 7                 |
| K                | Travel time                          | h       | Δ t 2 1 ሺ െ X ⁄ | ሻ ൏ K ൏ Δ  T 2X ⁄ |
| X                | Weighting coefficient                | -       | 0              | 0.5               |

Several trials are made to adapt the model structure following an iterative calibration process to understand better which hydrologic process needs to be improved in the model structure. In the iterative process, we either modify the conceptualization of the processes or change the parameter range of values used in the calibration process using OAT sensi -tivity analysis (see Section 3.2).
