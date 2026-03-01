package com.pricepredictor.connector;

/**
 * Base exception for all connector errors.
 */
public class PricePredictorException extends Exception {

    public PricePredictorException(String message) {
        super(message);
    }

    public PricePredictorException(String message, Throwable cause) {
        super(message, cause);
    }
}
