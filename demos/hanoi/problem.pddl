;; Five discs, three pegs. The optimal plan is exactly 2^5 - 1 = 31 moves,
;; which makes it a perfect ground truth for checking optimal planners.
(define (problem hanoi-5)
  (:domain hanoi)
  (:objects peg1 peg2 peg3 d1 d2 d3 d4 d5)

  (:init
    ;; every disc is smaller than every peg
    (smaller peg1 d1) (smaller peg1 d2) (smaller peg1 d3)
    (smaller peg1 d4) (smaller peg1 d5)
    (smaller peg2 d1) (smaller peg2 d2) (smaller peg2 d3)
    (smaller peg2 d4) (smaller peg2 d5)
    (smaller peg3 d1) (smaller peg3 d2) (smaller peg3 d3)
    (smaller peg3 d4) (smaller peg3 d5)
    ;; d1 is the smallest, d5 the largest
    (smaller d2 d1) (smaller d3 d1) (smaller d4 d1) (smaller d5 d1)
    (smaller d3 d2) (smaller d4 d2) (smaller d5 d2)
    (smaller d4 d3) (smaller d5 d3)
    (smaller d5 d4)

    (clear peg2) (clear peg3) (clear d1)
    (on d5 peg1) (on d4 d5) (on d3 d4) (on d2 d3) (on d1 d2))

  (:goal (and
    (on d5 peg3)
    (on d4 d5)
    (on d3 d4)
    (on d2 d3)
    (on d1 d2))))
