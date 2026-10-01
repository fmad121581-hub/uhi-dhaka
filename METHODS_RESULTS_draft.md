# Surface urban heat island of Dhaka from Landsat 8/9: methods and results (draft)

Working draft for the paper or portfolio write-up. Numbers come from `data/output_v2/`.

## Data
Landsat 8/9 Collection 2 Level-2 scenes (path 137, row 44) for five dates with at least 60% clear sky over the study area: 24 Feb 2022, 7 May 2025, 22 Oct 2025, 19 Feb 2026 and 3 Jun 2026. Surface temperature comes from ST_B10, surface reflectance from SR_B4 and SR_B5, cloud and shadow from QA_PIXEL. Land cover is ESA WorldCover 2021 (10 m, two tiles mosaicked because the study area crosses 24 N, collapsed to four classes), population is WorldPop 2020 (100 m), elevation is SRTM 30 m, and ward boundaries are GADM level 4 (203 units clipped to the Dhaka study boundary). All layers were resampled to a common 30 m UTM 45N grid, with nearest-neighbour resampling for categorical and integer layers.

## Methods
LST was taken directly from ST_B10 (0.00341802 x DN + 149 K). No further emissivity correction was applied, because the product is already emissivity-corrected. Fill pixels and values outside 0-72 C were removed. QA_PIXEL cloud, shadow, cirrus and snow flags were grown by five pixels (150 m) to catch thin cloud edges; dates with less than 60% clear pixels were dropped. NDVI was computed from reflectance (2.75e-5 x DN - 0.2). Water pixels were excluded from every statistic, since they are cool in daytime and have NDVI at or below zero.

SUHI intensity is the mean LST of dense built-up land (built-up pixel with at least 50% built-up cover within about 1 km) minus the mean LST of rural reference land (vegetation with at most 3% built-up cover within 1 km). For the single-date analysis a 95% interval came from a spatial block bootstrap (3 km blocks, 2000 resamples). Ward spatial structure was tested with global Moran's I and Getis-Ord Gi* (queen contiguity, 999 permutations). A Random Forest (150 trees) predicted pixel LST from NDVI, 330 m and 990 m built-up fraction, distance to water, elevation, population density and distance from the centre. It was validated with 3 km spatial block cross-validation, and random k-fold is reported next to it to show how much random splits overstate skill. Persistent hotspots are wards in the top LST decile on every usable date.

## Results
**Surface heat island.** On 24 Feb 2022 the urban core was 2.9 C warmer than the rural reference (95% CI 2.6 to 3.1). Across the five dates SUHI ranged from 2.9 C (Feb 2022) to 7.1 C (3 Jun 2026, monsoon onset), with 4.2 to 4.7 C in May, October and February 2026. A single scene therefore cannot define the heat island; the intensity depends on season and antecedent conditions. Mean land LST rose from 28.3 C in the February scene to 39.3 C in early June. Mid-monsoon (July to August) scenes were fully or mostly cloud covered, so the wettest weeks are not observable.

**Vegetation.** Over land pixels NDVI and LST were negatively correlated on four of five dates (r = -0.59 to -0.65); the post-monsoon date was weaker (r = -0.39). Including water pixels collapses the all-pixel correlation to about -0.1, which is why the pooled estimate in the first version of this analysis looked implausibly weak. LST fell by about 3.7 C from NDVI 0 to 0.2 to NDVI above 0.7 in the February 2022 scene.

**Spatial structure.** Ward mean LST is strongly clustered (Moran's I = 0.84, p = 0.001; 196 wards with enough land pixels). Gi* identified 45 hot and 45 cold wards at p < 0.05, with the hot cluster in the southern core around Old Dhaka, Shyampur and Kadamtali. LST falls from about 30.4 C within 6 km of the centre to 26.9 C beyond 40 km.

**Drivers.** The Random Forest explained R2 = 0.65 under spatial block cross-validation and 0.75 under random k-fold, so random splitting overstates skill by about 0.10. Permutation importance ranked distance from the centre (0.43), local built-up fraction at 330 m (0.18) and NDVI (0.12) highest. Distance from the centre partly stands in for urban form variables that were not measured (building height, surface albedo, anthropogenic heat), so it should not be read as a causal driver.

**Exposure.** About 48% of the study-area population lives in pixels at least one standard deviation above the mean LST on 24 Feb 2022 (30.0 C), and 25% in the hottest decile (30.7 C or above).

**Persistent hotspots.** Ward rankings are stable across dates (Spearman 0.74 to 0.89). Three wards are in the top decile on all five dates: Ward 90 (Shyampur), Ward 60 (Lalbagh) and Sultanganj Kamrangir Char. They are 81 to 99% built-up, hold about 117,000 people, and reach ward-mean LST of 46 to 47 C on 3 Jun 2026. Mean built-up share is 91% in persistent hotspots against 56% in other wards.

## Limitations to state
- Daytime (about 10:30 local) surface temperature, not air temperature. Night-time and canopy-layer heat islands can differ.
- Five dates over four and a half years, with no mid-monsoon coverage; seasonal means are not estimated.
- The Feb 2022 exposure and driver numbers come from one date and should be repeated on the other dates before publication.
- No ground validation yet. Matching LST against Bangladesh Meteorological Department station temperature on the scene dates would be the next check.
- Rainfall (NASA POWER, one grid point) shows a positive 2004-2023 trend (+55 mm/yr, p = 0.04) that depends on a single extreme year (2017); a rank-based test gives p = 0.055. Treat it as context, not a finding.
- Population is a 2020 model-based estimate redistributed over land cover, not a census count.
