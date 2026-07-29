;; Market day -- a showcase for :timed-initial-literals.
;;
;; The stalls open and close on a clock that runs whether or not the planner is
;; doing anything. Baking bread takes three hours; if you start too late, the
;; market has shut by the time you could sell it. Waiting is a legitimate move,
;; and the plan's makespan counts the idle time.
;;
;; Note what the sequential compilation costs you here: actions never overlap,
;; so you cannot bake one loaf while selling another. See jupyddl.requirements.
(define (domain market)
  (:requirements :strips :typing :durative-actions :timed-initial-literals
                 :numeric-fluents :adl)
  (:types goods)

  (:predicates
    (market-open)
    (baked ?g - goods)
    (sold ?g - goods))

  (:durative-action bake
    :parameters (?g - goods)
    :duration (= ?duration 3)
    :condition (and (at start (not (baked ?g))))
    :effect (and (at end (baked ?g))))

  (:durative-action sell
    :parameters (?g - goods)
    :duration (= ?duration 1)
    :condition (and (over all (market-open)) (over all (baked ?g)))
    :effect (and (at end (sold ?g)))))
