package com.pricepredictor.connector;

import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ResponseParsingTest {

    @Test
    void parseValidSuccessJson() throws Exception {
        String json = "{\"predicted_price_eur\": 2.35, \"model_version\": \"20260301-143000\"}";
        HttpServer server = MockHttpServer.responding(200, json);
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
        HttpServer server = MockHttpServer.responding(400, json);
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
        HttpServer server = MockHttpServer.responding(200, json);
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
}
