package com.pricepredictor.connector;

import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.ServerSocket;

import static org.junit.jupiter.api.Assertions.*;

class GracefulDegradationTest {

    private static final String OK_JSON = "{\"predicted_price_eur\": 1.0, \"model_version\": \"v1\"}";

    // --- T021: Timeout and connection-failure tests ---

    @Test
    void connectionToClosedPortThrowsServiceUnavailableWithin6Seconds() {
        // Find a port that is guaranteed closed
        int closedPort = findAvailablePort();
        var client = new PricePredictorClient("http://localhost:" + closedPort);
        var card = CardAttributes.builder().type("Instant").build();

        long start = System.currentTimeMillis();
        var ex = assertThrows(ServiceUnavailableException.class, () -> client.predict(card));
        long elapsed = System.currentTimeMillis() - start;

        assertTrue(elapsed < 6000, "Should fail within 6 seconds but took " + elapsed + "ms");
        assertNotNull(ex.getMessage());
    }

    @Test
    void slowServerTriggersTimeout() throws Exception {
        HttpServer server = MockHttpServer.respondingSlowly(10_000, 200, OK_JSON);
        server.start();
        int port = server.getAddress().getPort();
        try {
            // Client with 1 second timeout
            var client = new PricePredictorClient("http://localhost:" + port, 1000);
            var card = CardAttributes.builder().type("Instant").build();

            long start = System.currentTimeMillis();
            var ex = assertThrows(ServiceUnavailableException.class, () -> client.predict(card));
            long elapsed = System.currentTimeMillis() - start;

            assertTrue(elapsed < 3000, "Timeout should fire within ~1s but took " + elapsed + "ms");
            assertTrue(ex.getMessage().contains("timed out") || ex.getMessage().contains("Timeout")
                    || ex.getMessage().toLowerCase().contains("timeout"),
                    "Message should mention timeout: " + ex.getMessage());
        } finally {
            server.stop(0);
        }
    }

    @Test
    void configurableTimeoutIsRespected() throws Exception {
        HttpServer server = MockHttpServer.respondingSlowly(3000, 200, OK_JSON);
        server.start();
        int port = server.getAddress().getPort();
        try {
            // Client with 1 second timeout — should fail before server responds
            var client = new PricePredictorClient("http://localhost:" + port, 1000);
            var card = CardAttributes.builder().type("Instant").build();

            long start = System.currentTimeMillis();
            assertThrows(ServiceUnavailableException.class, () -> client.predict(card));
            long elapsed = System.currentTimeMillis() - start;

            assertTrue(elapsed < 2500, "1s timeout should trigger before 3s server delay, took " + elapsed + "ms");
        } finally {
            server.stop(0);
        }
    }

    @Test
    void exceptionMessageContainsUsefulContext() {
        int closedPort = findAvailablePort();
        var client = new PricePredictorClient("http://localhost:" + closedPort);
        var card = CardAttributes.builder().type("Instant").build();

        var ex = assertThrows(ServiceUnavailableException.class, () -> client.predict(card));
        String msg = ex.getMessage();
        assertTrue(msg.contains("localhost") && msg.contains(String.valueOf(closedPort)),
                "Message should contain URL info: " + msg);
    }

    // --- T022: Recovery tests ---

    @Test
    void clientRecoversAfterServiceRestart() throws Exception {
        // 1. Start mock server
        HttpServer server1 = MockHttpServer.responding(200, OK_JSON);
        server1.start();
        int port = server1.getAddress().getPort();

        var client = new PricePredictorClient("http://localhost:" + port, 2000);
        var card = CardAttributes.builder().type("Instant").build();

        // 2. Client succeeds
        PriceEstimate estimate = client.predict(card);
        assertEquals(1.0, estimate.predictedPriceEur(), 0.001);

        // 3. Stop server
        server1.stop(0);

        // 4. Client fails
        assertThrows(ServiceUnavailableException.class, () -> client.predict(card));

        // 5. Restart server on same port
        HttpServer server2 = MockHttpServer.respondingOnPort(port, 200, OK_JSON);
        server2.start();
        try {
            // 6. Client succeeds again — no recreation needed
            PriceEstimate recovered = client.predict(card);
            assertEquals(1.0, recovered.predictedPriceEur(), 0.001);
        } finally {
            server2.stop(0);
        }
    }

    @Test
    void clientDoesNotCacheConnectionState() throws Exception {
        HttpServer server = MockHttpServer.responding(200, OK_JSON);
        server.start();
        int port = server.getAddress().getPort();

        var client = new PricePredictorClient("http://localhost:" + port, 2000);
        var card = CardAttributes.builder().type("Instant").build();

        // Multiple successful calls
        client.predict(card);
        client.predict(card);

        // Stop and restart
        server.stop(0);
        HttpServer server2 = MockHttpServer.respondingOnPort(port, 200, OK_JSON);
        server2.start();
        try {
            // Still works after restart
            PriceEstimate result = client.predict(card);
            assertNotNull(result);
        } finally {
            server2.stop(0);
        }
    }

    // --- Helper ---

    private int findAvailablePort() {
        try (ServerSocket ss = new ServerSocket(0)) {
            return ss.getLocalPort();
        } catch (IOException e) {
            return 19999; // fallback
        }
    }
}
