;; Workshop -- exercises durative actions (sequential compilation).
(define (domain workshop)
  (:requirements :strips :typing :durative-actions)
  (:types part - object)
  (:predicates
    (cut ?p - part)
    (drilled ?p - part)
    (painted ?p - part)
    (finished ?p - part))

  (:durative-action cut
    :parameters (?p - part)
    :duration (= ?duration 2)
    :condition (and (at start (not (cut ?p))))
    :effect (and (at end (cut ?p))))

  (:durative-action drill
    :parameters (?p - part)
    :duration (= ?duration 3)
    :condition (and (over all (cut ?p)))
    :effect (and (at end (drilled ?p))))

  (:durative-action paint
    :parameters (?p - part)
    :duration (= ?duration 5)
    :condition (and (over all (drilled ?p)))
    :effect (and (at end (painted ?p))))

  (:durative-action inspect
    :parameters (?p - part)
    :duration (= ?duration 1)
    :condition (and (over all (painted ?p)))
    :effect (and (at end (finished ?p)))))
