package com.pricepredictor.connector;

import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;

import static org.junit.jupiter.api.Assertions.*;

class PricePredictorClientTest {

    @Test
    void successfulPredictionReturnsPriceEstimate() throws Exception {
        String json = "{\"predicted_price_eur\": 1.50, \"model_version\": \"test-v1\"}";
        HttpServer server = createMockServer(200, json, 0);
        server.start();
        try {
            var client = new PricePredictorClient(
                    "http://localhost:" + server.getAddress().getPort());
            var card = CardAttributes.builder()
                    .name("Test Card")
                    .types("Instant")
                    .manaCost("R")
                    .build();

            PriceEstimate estimate = client.predict(card);
            assertEquals(1.50, estimate.predictedPriceEur(), 0.001);
            assertEquals("test-v1", estimate.modelVersion());
        } finally {
            server.stop(0);
        }
    }

    @Test
    void server400ThrowsInvalidResponseException() throws Exception {
        String json = "{\"error\": \"Bad input\"}";
        HttpServer server = createMockServer(400, json, 0);
        server.start();
        try {
            var client = new PricePredictorClient(
                    "http://localhost:" + server.getAddress().getPort());
            var card = CardAttributes.builder().types("Instant").build();

            var ex = assertThrows(InvalidResponseException.class, () ->
                    client.predict(card));
            assertTrue(ex.getMessage().contains("Bad input"));
        } finally {
            server.stop(0);
        }
    }

    @Test
    void server500ThrowsInvalidResponseException() throws Exception {
        String json = "{\"error\": \"Internal error\"}";
        HttpServer server = createMockServer(500, json, 0);
        server.start();
        try {
            var client = new PricePredictorClient(
                    "http://localhost:" + server.getAddress().getPort());
            var card = CardAttributes.builder().types("Instant").build();

            assertThrows(InvalidResponseException.class, () -> client.predict(card));
        } finally {
            server.stop(0);
        }
    }

    @Test
    void requestSendsCorrectContentTypeAndMethod() throws Exception {
        final String[] capturedContentType = {null};
        final String[] capturedMethod = {null};

        HttpServer server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/api/v1/evaluate", exchange -> {
            capturedContentType[0] = exchange.getRequestHeaders().getFirst("Content-Type");
            capturedMethod[0] = exchange.getRequestMethod();
            exchange.getRequestBody().readAllBytes();
            String json = "{\"predicted_price_eur\": 1.0, \"model_version\": \"v1\"}";
            byte[] bytes = json.getBytes();
            exchange.sendResponseHeaders(200, bytes.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(bytes);
            }
        });
        server.start();
        try {
            var client = new PricePredictorClient(
                    "http://localhost:" + server.getAddress().getPort());
            client.predict(CardAttributes.builder().types("Instant").build());

            assertEquals("text/plain", capturedContentType[0]);
            assertEquals("POST", capturedMethod[0]);
        } finally {
            server.stop(0);
        }
    }

    private HttpServer createMockServer(int statusCode, String body, int delayMs) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/api/v1/evaluate", exchange -> {
            if (delayMs > 0) {
                try { Thread.sleep(delayMs); } catch (InterruptedException ignored) {}
            }
            exchange.getRequestBody().readAllBytes();
            byte[] bytes = body.getBytes();
            exchange.sendResponseHeaders(statusCode, bytes.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(bytes);
            }
        });
        return server;
    }
}
