from dataclasses import dataclass


@dataclass
class PromptWithVideo:
    text: str
    url: str


BREAKOUT = PromptWithVideo(
    text="""
    - Environment: You move a paddle and hit the ball in a brick wall at
        the top of the screen. Your goal is to destroy the brick wall.
        Each time the ball hits a brick, the brick disappears and you
        score points.
    - Reward: You score points by destroying bricks in the wall. The
        reward for destroying a brick depends on the color of the brick.
        Red - 7 points
        Orange - 7 points
        Yellow - 4 points
        Green - 4 points
        Aqua - 1 point
        Blue - 1 point
    - Actions: The possible actions are moving the paddle to the left
        or to the right.
    """,
    url="https://www.youtube.com/shorts/mh98w3auWMw",
)

SPACE_INVADERS = PromptWithVideo(
    text="""
    - Environment: Each time you turn on SPACE INVADERS you will be at war with
        enemies from space who are threatening the earth. Your
        objective is to destroy these invaders by firing your "laser
        cannon." You must wipe out the invaders either before they
        reach the earth (bottom of the screen), or before they hit you
        three times with their "laser bombs."
        The SPACE INVADERS move faster on the screen as their numbers
        decrease, making them more difficult to hit. The fastest speed
        occurs when only one invader remains on the screen.
        There are SHIELDS (screen diagram) positioned on the screen
        between your laser cannon and the SPACE INVADERS. At the
        outset you are safe behind the SHIELDS. However, as you and
        the enemy hit the SHIELDS, they become damaged, allowing laser
        beams from your cannon and laser bombs from the enemy to pass
        through them. As the SPACE INVADERS get close to the SHIELDS
        on their way to the earth, the SHIELDS disappear altogether.
    - Reward: Your long-term objective is to score as many points as
        possible. Points are scored each time you hit one of the SPACE
        INVADERS. They are worth different amounts of points,
        depending on their initial position on the screen.
        The SPACE INVADERS are worth 5, 10, 15, 20, 25, 30 points in
        the first through sixth rows respectively. The point value of
        each target stays the same as it drops lower on the screen.
        Each complete set of SPACE INVADERS is worth 630
        points.
    - Actions: moving your ship to the left, right, or firing your cannon.
    """,
    url="https://www.youtube.com/shorts/J6_kEGPib50",
)
