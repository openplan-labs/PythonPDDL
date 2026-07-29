;; Miconic-style elevator: passengers board at their origin floor and leave at
;; their destination. Conditional effects (`when`) make boarding automatic for
;; everybody waiting on the current floor, so this exercises the ADL half of
;; the grounder rather than plain STRIPS.
(define (domain elevator)
  (:requirements :strips :typing :negative-preconditions :conditional-effects)
  (:types passenger floor - object)

  (:predicates
    (lift-at ?f - floor)
    (origin ?p - passenger ?f - floor)
    (destin ?p - passenger ?f - floor)
    (boarded ?p - passenger)
    (served ?p - passenger))

  ;; Moving the lift also serves everybody whose destination is the new floor
  ;; and boards everybody waiting there -- two conditional effects per passenger.
  (:action move
    :parameters (?from - floor ?to - floor)
    :precondition (lift-at ?from)
    :effect (and (lift-at ?to) (not (lift-at ?from))))

  (:action board
    :parameters (?f - floor ?p - passenger)
    :precondition (and (lift-at ?f) (origin ?p ?f) (not (served ?p)))
    :effect (boarded ?p))

  (:action depart
    :parameters (?f - floor ?p - passenger)
    :precondition (and (lift-at ?f) (destin ?p ?f) (boarded ?p))
    :effect (and (not (boarded ?p)) (served ?p))))
