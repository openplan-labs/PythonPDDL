;; Errands -- a showcase for PDDL 3: soft goals and trajectory constraints.
;;
;; The hard requirement is only to get home. Everything else is expressed as
;; things you would *rather* were true, priced by the metric, plus rules the
;; whole trip must obey:
;;
;;   (always (or (at-home) (has-keys)))  -- never leave without your keys
;;   (sometime-before (locked) (at-home)) -- you can only lock up from inside
;;
;; A planner that ignored the preferences would still produce a legal plan; it
;; would just be a worse one. That is the difference between a constraint and a
;; preference, and this domain exists to make it visible.
(define (domain errands)
  (:requirements :adl :preferences :constraints :action-costs)

  (:predicates
    (at-home)
    (has-keys)
    (locked)
    (bought-milk)
    (bought-bread)
    (posted-letter))

  (:functions (total-cost))

  (:action take-keys
    :precondition (and (at-home) (not (has-keys)))
    :effect (and (has-keys) (increase (total-cost) 1)))

  (:action lock-up
    :precondition (and (at-home) (has-keys) (not (locked)))
    :effect (and (locked) (increase (total-cost) 1)))

  (:action go-out
    :precondition (at-home)
    :effect (and (not (at-home)) (increase (total-cost) 2)))

  (:action come-home
    :precondition (not (at-home))
    :effect (and (at-home) (increase (total-cost) 2)))

  (:action buy-milk
    :precondition (not (at-home))
    :effect (and (bought-milk) (increase (total-cost) 1)))

  (:action buy-bread
    :precondition (not (at-home))
    :effect (and (bought-bread) (increase (total-cost) 1)))

  (:action post-letter
    :precondition (not (at-home))
    :effect (and (posted-letter) (increase (total-cost) 3))))
