;; The market runs from 08:00 to 12:00. Baking takes three hours, so the loaf
;; is ready at 03:00 -- five hours before anyone can buy it. The planner has to
;; wait for the market to open, and the resulting makespan (9) is dominated by
;; that idle time rather than by the work.
(define (problem market-day)
  (:domain market)
  (:objects bread - goods)

  (:init
    (at 8 (market-open))
    (at 12 (not (market-open))))

  (:goal (sold bread)))
