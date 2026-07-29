;; Six balls to carry from room A to room B, two grippers.
;; Optimal plan: 17 actions. Blind search explodes; h_FF walks straight in.
(define (problem gripper-6)
  (:domain gripper)
  (:objects
    rooma roomb - room
    ball1 ball2 ball3 ball4 ball5 ball6 - ball
    left right - gripper)

  (:init
    (at-robby rooma)
    (free left)
    (free right)
    (at ball1 rooma)
    (at ball2 rooma)
    (at ball3 rooma)
    (at ball4 rooma)
    (at ball5 rooma)
    (at ball6 rooma))

  (:goal (and
    (at ball1 roomb)
    (at ball2 roomb)
    (at ball3 roomb)
    (at ball4 roomb)
    (at ball5 roomb)
    (at ball6 roomb))))
