;; Transport with fuel -- exercises numeric fluents.
(define (domain numeric-transport)
  (:requirements :strips :typing :numeric-fluents)
  (:types truck location package - object)
  (:predicates
    (at ?t - truck ?l - location)
    (road ?a ?b - location)
    (carrying ?p - package ?t - truck)
    (package-at ?p - package ?l - location))
  (:functions
    (fuel ?t - truck)
    (distance ?a ?b - location))

  (:action drive
    :parameters (?t - truck ?from ?to - location)
    :precondition (and (at ?t ?from) (road ?from ?to)
                       (>= (fuel ?t) (distance ?from ?to)))
    :effect (and (not (at ?t ?from)) (at ?t ?to)
                 (decrease (fuel ?t) (distance ?from ?to))))

  (:action refuel
    :parameters (?t - truck)
    :precondition (< (fuel ?t) 40)
    :effect (assign (fuel ?t) 60))

  (:action load
    :parameters (?p - package ?t - truck ?l - location)
    :precondition (and (at ?t ?l) (package-at ?p ?l))
    :effect (and (not (package-at ?p ?l)) (carrying ?p ?t)))

  (:action unload
    :parameters (?p - package ?t - truck ?l - location)
    :precondition (and (at ?t ?l) (carrying ?p ?t))
    :effect (and (not (carrying ?p ?t)) (package-at ?p ?l))))
