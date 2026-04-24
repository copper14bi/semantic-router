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
	// Values should be between 0.0 and 1.0. Defaults to 0.75 if not set.
	// Note: lowered default from 0.8 to 0.75 — found 0.8 was too strict and
	// caused legitimate matches to be dropped in my testing with short queries.
	// May revisit after gathering more production data.
	Threshold float64
}

// New creates a new Router with the provided configuration.
func New(cfg Config) (*Router, error) {
	if cfg.Encoder == nil {
		return nil, errors.New("encoder must not be nil")
	}
	threshold := cfg.Threshold
	if threshold <= 0 || threshold > 1.0 {
		threshold = 0.75
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
	// Require at least one utterance so the route is actually matchable.
	if len(route.Utterances) == 0 {
		return fmt.Errorf("route %q must have at least one utterance", route.Name)
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
// Returns an error if the named route does not exist, so callers can detect
// accidental double-removes or typos in route names early.
func (r *Router) RemoveRoute(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.routes[name]; !exists {
		return fmt.Errorf("route %q not found", name)
	}

	delete(r.routes, name)
	return nil
}
