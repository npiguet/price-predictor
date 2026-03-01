package com.pricepredictor.connector;

/**
 * Thrown when the server returns a non-200 status or an unparseable response.
 */
public class InvalidResponseException extends PricePredictorException {

    public InvalidResponseException(String message) {
        super(message);
    }

    public InvalidResponseException(String message, Throwable cause) {
        super(message, cause);
    }
}
