// Package router provides the core semantic routing functionality.
// It routes incoming requests to the appropriate backend based on
// semantic similarity of the request content to configured routes.
package router

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/pkg/types"
)

// ErrNoRouteFound is returned when no matching route is found for a request.
var ErrNoRouteFound = errors.New("no matching route found")

// ErrRouteAlreadyExists is returned when attempting to register a duplicate route.
var ErrRouteAlreadyExists = errors.New("route already exists")

// Router is the main semantic router that matches requests to routes
// based on semantic similarity scoring.
type Router struct {
	mu       sync.RWMutex
	routes   map[string]*types.Route
	encoder  types.Encoder
	threshold float64
}

// Config holds the configuration for creating a new Router.
type Config struct {
	// Encoder is used to generate embeddings for semantic comparison.
	Encoder types.Encoder
	// Threshold is the minimum similarity score required to match a route.
	// Values should be between 0.0 and 1.0. Defaults to 0.7 if not set.
	Threshold float64
}

// New creates a new Router with the provided configuration.
func New(cfg Config) (*Router, error) {
	if cfg.Encoder == nil {
		return nil, errors.New("encoder must not be nil")
	}
	threshold := cfg.Threshold
	if threshold <= 0 || threshold > 1.0 {
		threshold = 0.7
	}
	return &Router{
		routes:    make(map[string]*types.Route),
		encoder:   cfg.Encoder,
		threshold: threshold,
	}, nil
}

// AddRoute registers a new route with the router.
// Returns ErrRouteAlreadyExists if a route with the same name is already registered.
func (r *Router) AddRoute(ctx context.Context, route *types.Route) error {
	if route == nil {
		return errors.New("route must not be nil")
	}
	if route.Name == "" {
		return errors.New("route name must not be empty")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.routes[route.Name]; exists {
		return fmt.Errorf("%w: %s", ErrRouteAlreadyExists, route.Name)
	}

	// Pre-compute embeddings for all utterances in this route.
	for i, utterance := range route.Utterances {
		embedding, err := r.encoder.Encode(ctx, utterance)
		if err != nil {
			return fmt.Errorf("failed to encode utterance %d for route %s: %w", i, route.Name, err)
		}
		route.Embeddings = append(route.Embeddings, embedding)
	}

	r.routes[route.Name] = route
	return nil
}

// RemoveRoute removes a registered route by name.
func (r *Router) RemoveRoute(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.routes[name]; !exists {
		return fmt.Errorf("route %q not found", name)
	}
	delete(r.routes, name)
	return nil
}

// Match finds the best matching route for the given query string.
// Returns ErrNoRouteFound if no route meets the similarity threshold.
func (r *Router) Match(ctx context.Context, query string) (*types.RouteMatch, error) {
	queryEmbedding, err := r.encoder.Encode(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to encode query: %w", err)
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	var (
		bestRoute *types.Route
		bestScore float64
	)

	for _, route := range r.routes {
		for _, embedding := range route.Embeddings {
			score := cosineSimilarity(queryEmbedding, embedding)
			if score > bestScore {
				bestScore = score
				bestRoute = route
			}
		}
	}

	if bestRoute == nil || bestScore < r.threshold {
		return nil, ErrNoRouteFound
	}

	return &types.RouteMatch{
		Route: bestRoute,
		Score: bestScore,
	}, nil
}

// cosineSimilarity computes the cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (sqrt(normA) * sqrt(normB))
}

// sqrt is a simple square root helper to avoid importing math in hot path.
func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 50; i++ {
		z -= (z*z - x) / (2 * z)
	}
	return z
}
