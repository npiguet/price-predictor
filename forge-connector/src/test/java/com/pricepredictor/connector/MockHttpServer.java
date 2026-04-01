package com.pricepredictor.connector;

import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;

/**
 * Test helper for creating lightweight mock HTTP servers that respond to the evaluate endpoint.
 */
class MockHttpServer {

    static final String EVALUATE_PATH = "/api/v1/evaluate";

    /** Create a server on a random port that responds immediately with the given status and body. */
    static HttpServer responding(int statusCode, String body) throws IOException {
        return respondingOnPort(0, statusCode, body);
    }

    /** Create a server on a specific port that responds immediately with the given status and body. */
    static HttpServer respondingOnPort(int port, int statusCode, String body) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        byte[] bytes = body.getBytes();
        server.createContext(EVALUATE_PATH, exchange -> {
            exchange.getRequestBody().readAllBytes();
            exchange.sendResponseHeaders(statusCode, bytes.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(bytes);
            }
        });
        return server;
    }

    /** Create a server that sleeps {@code delayMs} before responding. */
    static HttpServer respondingSlowly(int delayMs, int statusCode, String body) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(0), 0);
        byte[] bytes = body.getBytes();
        server.createContext(EVALUATE_PATH, exchange -> {
            try { Thread.sleep(delayMs); } catch (InterruptedException ignored) {}
            exchange.getRequestBody().readAllBytes();
            exchange.sendResponseHeaders(statusCode, bytes.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(bytes);
            }
        });
        return server;
    }
}
