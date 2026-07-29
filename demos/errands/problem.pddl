;; Get home, ideally having done the errands -- and never having left the house
;; without your keys.
;;
;; The metric prices each missed errand. Milk is worth 6 and bread 6, both far
;; more than the 1 they cost to buy, so a cost-optimal planner will fetch them.
;; Posting the letter costs 3 but is only worth 2, so the optimal plan
;; deliberately *skips* it and pays the penalty -- which is exactly the answer a
;; soft goal is supposed to allow.
(define (problem errands-day)
  (:domain errands)

  (:init
    (at-home)
    (= (total-cost) 0))

  (:constraints (and
    ;; Stepping outside without your keys is never allowed, at any point.
    (always (or (at-home) (has-keys)))
    ;; The door can only be locked from inside, so being home must come first.
    (sometime-before (locked) (at-home))))

  (:goal (and
    (at-home)
    (locked)
    (preference milk (bought-milk))
    (preference bread (bought-bread))
    (preference letter (posted-letter))))

  (:metric minimize (+ (total-cost)
                       (+ (* 6 (is-violated milk))
                          (+ (* 6 (is-violated bread))
                             (* 2 (is-violated letter)))))))
