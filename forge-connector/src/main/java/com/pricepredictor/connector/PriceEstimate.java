package com.pricepredictor.connector;

/**
 * Prediction result containing the estimated EUR price and model version.
 */
public record PriceEstimate(
    double predictedPriceEur,
    String modelVersion
) {}
