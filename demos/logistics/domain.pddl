;; Logistics: trucks move packages inside a city, airplanes move them between
;; airports. Action costs make flying expensive, so cost-optimal planners have
;; to reason about *which* route, not just how many steps.
(define (domain logistics)
  (:requirements :strips :typing :action-costs)
  (:types
    truck airplane - vehicle
    package vehicle - thing
    airport location - place
    city - object)

  (:predicates
    (at ?t - thing ?p - place)
    (in ?p - package ?v - vehicle)
    (in-city ?p - place ?c - city))

  (:functions (total-cost))

  (:action load
    :parameters (?p - package ?v - vehicle ?l - place)
    :precondition (and (at ?p ?l) (at ?v ?l))
    :effect (and (not (at ?p ?l)) (in ?p ?v)
                 (increase (total-cost) 1)))

  (:action unload
    :parameters (?p - package ?v - vehicle ?l - place)
    :precondition (and (in ?p ?v) (at ?v ?l))
    :effect (and (not (in ?p ?v)) (at ?p ?l)
                 (increase (total-cost) 1)))

  (:action drive
    :parameters (?t - truck ?from - place ?to - place ?c - city)
    :precondition (and (at ?t ?from) (in-city ?from ?c) (in-city ?to ?c))
    :effect (and (not (at ?t ?from)) (at ?t ?to)
                 (increase (total-cost) 2)))

  (:action fly
    :parameters (?a - airplane ?from - airport ?to - airport)
    :precondition (at ?a ?from)
    :effect (and (not (at ?a ?from)) (at ?a ?to)
                 (increase (total-cost) 6))))
