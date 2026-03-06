package com.pricepredictor.connector;

import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;

import static org.junit.jupiter.api.Assertions.*;

class ResponseParsingTest {

    @Test
    void parseValidSuccessJson() throws Exception {
        String json = "{\"predicted_price_eur\": 2.35, \"model_version\": \"20260301-143000\"}";
        HttpServer server = createMockServer(200, json);
        server.start();
        try {
            var client = new PricePredictorClient(
                    "http://localhost:" + server.getAddress().getPort());
            PriceEstimate estimate = client.predict(
                    CardAttributes.builder().type("Instant").build());
            assertEquals(2.35, estimate.predictedPriceEur(), 0.001);
            assertEquals("20260301-143000", estimate.modelVersion());
        } finally {
            server.stop(0);
        }
    }

    @Test
    void parseErrorJsonExtractsMessage() throws Exception {
        String json = "{\"error\": \"No Types line found\"}";
        HttpServer server = createMockServer(400, json);
        server.start();
        try {
            var client = new PricePredictorClient(
                    "http://localhost:" + server.getAddress().getPort());
            var ex = assertThrows(InvalidResponseException.class, () ->
                    client.predict(CardAttributes.builder().type("Instant").build()));
            assertTrue(ex.getMessage().contains("No Types line found"));
        } finally {
            server.stop(0);
        }
    }

    @Test
    void malformedJsonThrowsInvalidResponseException() throws Exception {
        String json = "this is not json at all";
        HttpServer server = createMockServer(200, json);
        server.start();
        try {
            var client = new PricePredictorClient(
                    "http://localhost:" + server.getAddress().getPort());
            assertThrows(InvalidResponseException.class, () ->
                    client.predict(CardAttributes.builder().type("Instant").build()));
        } finally {
            server.stop(0);
        }
    }

    private HttpServer createMockServer(int statusCode, String body) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/api/v1/evaluate", exchange -> {
            // Read request body to avoid broken pipe
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
