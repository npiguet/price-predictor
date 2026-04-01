package com.pricepredictor.connector;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.net.ConnectException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.time.Duration;

/**
 * Client for the card price prediction service.
 * Thread-safe and reusable across requests.
 */
public class PricePredictorClient {

    private static final String DEFAULT_ENDPOINT = "http://localhost:8000";
    private static final int DEFAULT_TIMEOUT_MS = 5000;

    private final String endpointUrl;
    private final int timeoutMs;
    private final HttpClient httpClient;

    public PricePredictorClient() {
        this(DEFAULT_ENDPOINT, DEFAULT_TIMEOUT_MS);
    }

    public PricePredictorClient(String endpointUrl) {
        this(endpointUrl, DEFAULT_TIMEOUT_MS);
    }

    public PricePredictorClient(String endpointUrl, int timeoutMs) {
        this.endpointUrl = endpointUrl;
        this.timeoutMs = timeoutMs;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(timeoutMs))
                .build();
    }

    /**
     * Predict the EUR price for a card.
     *
     * @param card the card attributes to predict
     * @return a PriceEstimate with the predicted price and model version
     * @throws ServiceUnavailableException if the service is unreachable or times out
     * @throws InvalidResponseException if the server returns an error or unparseable response
     */
    public PriceEstimate predict(CardAttributes card) throws PricePredictorException {
        String body = ForgeScriptSerializer.serialize(card);
        String url = endpointUrl + "/api/v1/evaluate";

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofMillis(timeoutMs))
                .header("Content-Type", "text/plain")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

        HttpResponse<String> response;
        try {
            response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        } catch (HttpTimeoutException e) {
            throw new ServiceUnavailableException(
                    "Request timed out after " + timeoutMs + "ms: " + url, e);
        } catch (ConnectException e) {
            throw new ServiceUnavailableException(
                    "Connection refused: " + url, e);
        } catch (IOException e) {
            throw new ServiceUnavailableException(
                    "Network error: " + e.getMessage() + " (" + url + ")", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ServiceUnavailableException(
                    "Request interrupted: " + url, e);
        }

        String responseBody = response.body();

        if (response.statusCode() == 200) {
            return parseSuccessResponse(responseBody);
        } else {
            String errorMsg = parseErrorMessage(responseBody);
            throw new InvalidResponseException(errorMsg);
        }
    }

    private PriceEstimate parseSuccessResponse(String json) throws InvalidResponseException {
        try {
            JsonObject obj = JsonParser.parseString(json).getAsJsonObject();
            double price = obj.get("predicted_price_eur").getAsDouble();
            String version = obj.get("model_version").getAsString();
            return new PriceEstimate(price, version);
        } catch (Exception e) {
            throw new InvalidResponseException("Failed to parse response: " + e.getMessage(), e);
        }
    }

    private String parseErrorMessage(String json) {
        try {
            return JsonParser.parseString(json).getAsJsonObject().get("error").getAsString();
        } catch (Exception e) {
            return "Server returned status error with unparseable body";
        }
    }
}
