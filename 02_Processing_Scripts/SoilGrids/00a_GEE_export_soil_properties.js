// =============================================================================
// 00a_GEE_export_soil_properties.js
//
// Google Earth Engine (JavaScript Code Editor) script.
// Computes the thickness-weighted mean over 0-200 cm of six SoilGrids v2.0
// soil properties (bulk density, coarse fragments, clay, sand, silt, SOC),
// clips them to the study area and exports them as GeoTIFFs (250 m,
// EPSG:4326) to Google Drive.
//
// Run in: https://code.earthengine.google.com  (not a Python script)
// After the export, copy the GeoTIFFs to 01_Input_Global/1_Original/SoilGrids/
// and run 01_soilgrids_build_nc.py.
// =============================================================================

// 1. Study area
var region = ee.Geometry.Rectangle([5.5, 47, 15.5, 55.5]);

// 2. Layer thicknesses (cm) used as weights
var weights = [5, 10, 15, 30, 40, 100];
var totalThickness = 200;

// Thickness-weighted mean over the six depth bands. The band prefix
// (e.g. "clay", "soc") is read from the first band name.
var calculateWeightedMean = function(image, conversionFactor) {
  // First band name
  var firstBand = ee.String(image.bandNames().get(0));

  // Prefix = everything before the first underscore (server-side operation)
  var prefix = firstBand.split('_').get(0);

  var depths = [
    '_0-5cm_mean',
    '_5-15cm_mean',
    '_15-30cm_mean',
    '_30-60cm_mean',
    '_60-100cm_mean',
    '_100-200cm_mean'
  ];

  var weightedSum = ee.Image(0);

  for (var i = 0; i < depths.length; i++) {
    // Band name must be built as an ee.String
    var bandName = ee.String(prefix).cat(depths[i]);

    var layer = image.select(bandName);
    weightedSum = weightedSum.add(layer.multiply(weights[i]));
  }

  return weightedSum.divide(totalThickness).divide(conversionFactor).float();
};

// 3. Load SoilGrids v2.0 assets
var bd_raw     = ee.Image("projects/soilgrids-isric/bdod_mean");
var cfvo_raw   = ee.Image("projects/soilgrids-isric/cfvo_mean");
var clay_raw   = ee.Image("projects/soilgrids-isric/clay_mean");
var sand_raw   = ee.Image("projects/soilgrids-isric/sand_mean");
var silt_raw   = ee.Image("projects/soilgrids-isric/silt_mean");
var soc_raw    = ee.Image("projects/soilgrids-isric/soc_mean");

// 4. Weighted mean and unit conversion (second argument = divisor)
var bd_final   = calculateWeightedMean(bd_raw, 100).clip(region);
var cfvo_final = calculateWeightedMean(cfvo_raw, 10).clip(region);
var clay_final = calculateWeightedMean(clay_raw, 10).clip(region);
var sand_final = calculateWeightedMean(sand_raw, 10).clip(region);
var silt_final = calculateWeightedMean(silt_raw, 10).clip(region);
var soc_final  = calculateWeightedMean(soc_raw, 10).clip(region);

// 5. Export to Google Drive
var exportImage = function(image, name, fileName) {
  Export.image.toDrive({
    image: image,
    description: name,
    folder: 'SoilGrids_Data_New',
    fileNamePrefix: fileName,
    scale: 250,
    region: region,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
};

exportImage(bd_final,   'SoilGrids_BulkDensity', 'BD_0-200cm_Mean');
exportImage(cfvo_final, 'SoilGrids_CoarseFragments', 'CF_0-200cm_Mean');
exportImage(clay_final, 'SoilGrids_Clay', 'Clay_0-200cm_Mean');
exportImage(sand_final, 'SoilGrids_Sand', 'Sand_0-200cm_Mean');
exportImage(silt_final, 'SoilGrids_Silt', 'Silt_0-200cm_Mean');
exportImage(soc_final,  'SoilGrids_SOC', 'SOC_0-200cm_Mean');

// 6. Visual check
Map.centerObject(region, 6);
Map.addLayer(clay_final, {min: 10, max: 40, palette: ['yellow', 'orange', 'red']}, 'Clay Content %');
