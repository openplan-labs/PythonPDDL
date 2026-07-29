;; Two towers in, one inverted tower out. Eight blocks is the sweet spot:
;; small enough to solve in a second, big enough that the choice of heuristic
;; changes the number of expansions by orders of magnitude.
(define (problem blocksworld-8)
  (:domain blocksworld)
  (:objects a b c d e f g h - block)

  (:init
    (handempty)
    (ontable a) (on b a) (on c b) (on d c) (clear d)
    (ontable e) (on f e) (on g f) (on h g) (clear h))

  (:goal (and
    (ontable h)
    (on g h)
    (on f g)
    (on e f)
    (on d e)
    (on c d)
    (on b c)
    (on a b))))
