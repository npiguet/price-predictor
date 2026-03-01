package com.pricepredictor.connector;

/**
 * Thrown when the prediction service is unreachable (connection refused, timeout).
 */
public class ServiceUnavailableException extends PricePredictorException {

    public ServiceUnavailableException(String message) {
        super(message);
    }

    public ServiceUnavailableException(String message, Throwable cause) {
        super(message, cause);
    }
}
