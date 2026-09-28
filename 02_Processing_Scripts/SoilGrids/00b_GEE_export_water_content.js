// =============================================================================
// 00b_GEE_export_water_content.js
//
// Google Earth Engine (JavaScript Code Editor) script.
// Computes the thickness-weighted mean over 0-200 cm of the SoilGrids v2.0
// volumetric water content at 10 kPa, 33 kPa and 1500 kPa, clips it to the
// study area and exports it as GeoTIFFs (250 m, EPSG:4326) to Google Drive.
//
// Run in: https://code.earthengine.google.com  (not a Python script)
// After the export, copy the GeoTIFFs to 01_Input_Global/1_Original/SoilGrids/
// and run 01_soilgrids_build_nc.py (which applies the x1000 VWC correction).
// =============================================================================

// 1. Study area
var region = ee.Geometry.Rectangle([5.5, 47, 15.5, 55.5]);

// 2. Layer thicknesses (cm) used as weights
var weights = [5, 10, 15, 30, 40, 100];
var totalThickness = 200;

// Thickness-weighted mean over the six depth bands (band names val_...)
var calculateWeightedMean = function(image) {
  var depths = [
    'val_0_5cm_mean',
    'val_5_15cm_mean',
    'val_15_30cm_mean',
    'val_30_60cm_mean',
    'val_60_100cm_mean',
    'val_100_200cm_mean'
  ];

  var weightedSum = ee.Image(0);

  for (var i = 0; i < depths.length; i++) {
    var bandName = depths[i];
    var layer = image.select(bandName);
    weightedSum = weightedSum.add(layer.multiply(weights[i]));
  }

  // Division by the total thickness (200 cm) and by 1000
  return weightedSum.divide(totalThickness).divide(1000).float();
};

// 3. Load SoilGrids v2.0 water content assets
var vwc10_raw = ee.Image("ISRIC/SoilGrids250m/v2_0/wv0010");
var vwc33_raw = ee.Image("ISRIC/SoilGrids250m/v2_0/wv0033");
var vwc1500_raw = ee.Image("ISRIC/SoilGrids250m/v2_0/wv1500");

// 4. Weighted mean
var vwc10_final = calculateWeightedMean(vwc10_raw).clip(region);
var vwc33_final = calculateWeightedMean(vwc33_raw).clip(region);
var vwc1500_final = calculateWeightedMean(vwc1500_raw).clip(region);

// 5. Export to Google Drive

// Export 10kPa
Export.image.toDrive({
  image: vwc10_final,
  description: 'SoilGrids_VWC_10kPa_WeightedMean_Germany',
  folder: 'SoilGrids_Data',
  fileNamePrefix: 'VWC_10kPa_0-200cm_Mean',
  scale: 250,
  region: region,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// Export 33kPa
Export.image.toDrive({
  image: vwc33_final,
  description: 'SoilGrids_VWC_33kPa_WeightedMean_Germany',
  folder: 'SoilGrids_Data',
  fileNamePrefix: 'VWC_33kPa_0-200cm_Mean',
  scale: 250,
  region: region,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// Export 1500kPa
Export.image.toDrive({
  image: vwc1500_final,
  description: 'SoilGrids_VWC_1500kPa_WeightedMean_Germany',
  folder: 'SoilGrids_Data',
  fileNamePrefix: 'VWC_1500kPa_0-200cm_Mean',
  scale: 250,
  region: region,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// Visual check
Map.centerObject(region, 6);
Map.addLayer(vwc33_final, {min: 0.1, max: 0.5, palette: ['brown', 'yellow', 'blue']}, 'VWC 33kPa Mean');
Map.addLayer(vwc1500_final, {min: 0.05, max: 0.3, palette: ['red', 'orange', 'green']}, 'VWC 1500kPa Mean');
