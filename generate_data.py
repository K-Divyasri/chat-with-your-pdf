"""Create the sample PDFs you will chat with. Run this once, first.

    python generate_data.py

It writes three small, multi-page PDFs into ./data:

    remote_work_policy.pdf   a fake company handbook -- the classic RAG demo
                             (HR Q&A: "how many vacation days?"). Answers are
                             specific and quotable, which is perfect for citations.
    solar_system.pdf         friendly planet facts -- a fun, universal document.
    coffee_guide.pdf         a short brewing guide -- a third topic so retrieval
                             has to actually pick the right document.

The content is written so that questions and answers share distinctive words,
which is what the offline lexical retriever needs to shine. Everything is made-up;
nothing here is real company policy or scientific authority.
"""

from __future__ import annotations

from pathlib import Path

from pdf_chat.simple_pdf import write_text_pdf

DATA_DIR = Path(__file__).parent / "data"


REMOTE_WORK_POLICY = [
    [
        "ACME WIDGETS -- REMOTE WORK POLICY",
        "Version 3, effective January 2026.",
        "",
        "This handbook explains how remote work operates at Acme Widgets. It",
        "covers eligibility, working hours, equipment, vacation and time off,",
        "expenses, and security. Keep this document handy; it is the source of",
        "truth for anything related to working from home.",
        "",
        "1. ELIGIBILITY",
        "All full-time employees who have completed their 90-day probation may",
        "work remotely up to four days per week. Fully remote arrangements need",
        "written approval from your manager and the head of your department.",
    ],
    [
        "2. WORKING HOURS",
        "Core collaboration hours are 10am to 3pm in your team's main time zone.",
        "You may arrange the rest of your working day flexibly. If you work",
        "across time zones, agree a shared overlap window with your manager.",
        "",
        "3. VACATION AND TIME OFF",
        "Full-time employees receive 25 days of paid vacation per year, plus",
        "public holidays. Vacation accrues monthly and unused days may be carried",
        "over up to a maximum of 5 days into the next year. Request vacation at",
        "least two weeks in advance through the HR portal. Sick leave is separate",
        "and is capped at 10 paid days per year.",
    ],
    [
        "4. EQUIPMENT AND EXPENSES",
        "Acme provides a laptop and a monitor to every remote employee. You may",
        "claim up to 500 dollars for a home-office desk and chair in your first",
        "year. Internet costs are reimbursed at 30 dollars per month. Submit all",
        "expense claims through the finance portal within 60 days.",
        "",
        "5. SECURITY",
        "Always connect through the company VPN when handling customer data. Turn",
        "on full-disk encryption and lock your screen when you step away. Report",
        "any lost device to the security team within 24 hours.",
    ],
]


SOLAR_SYSTEM = [
    [
        "A SHORT GUIDE TO THE SOLAR SYSTEM",
        "",
        "The solar system is the Sun and everything bound to it by gravity: eight",
        "planets, their moons, dwarf planets, asteroids, and comets. This guide",
        "gives you one friendly page on each of the more famous members.",
        "",
        "THE SUN",
        "The Sun is a star at the center of the solar system. It is a huge ball of",
        "hot plasma, and its gravity holds every planet in orbit. It provides the",
        "light and heat that make life on Earth possible.",
    ],
    [
        "MARS -- THE RED PLANET",
        "Mars is the fourth planet from the Sun. It looks red because its soil is",
        "rich in iron oxide, which is basically rust. Mars has two tiny moons,",
        "Phobos and Deimos, and the tallest volcano in the solar system, Olympus",
        "Mons. A day on Mars is about 24 hours and 39 minutes, close to Earth's.",
    ],
    [
        "JUPITER -- THE GIANT",
        "Jupiter is the largest planet in the solar system, so big that all the",
        "other planets could fit inside it. Its most famous feature is the Great",
        "Red Spot, a storm wider than Earth that has raged for centuries. Jupiter",
        "has 95 confirmed moons, including Ganymede, the largest moon in the",
        "solar system.",
    ],
    [
        "SATURN -- THE RINGED PLANET",
        "Saturn is best known for its bright rings, which are made of countless",
        "chunks of ice and rock. It is the second-largest planet. Saturn is a gas",
        "giant with no solid surface, and it is so light for its size that it",
        "would float in water if you could find a big enough bathtub.",
    ],
]


COFFEE_GUIDE = [
    [
        "THE SMALL COFFEE BREWING GUIDE",
        "",
        "Good coffee is mostly about three things: fresh beans, the right grind,",
        "and the ratio of coffee to water. This guide keeps it simple and covers",
        "two easy brewing methods you can do at home without fancy machines.",
        "",
        "THE GOLDEN RATIO",
        "A reliable starting point is 60 grams of coffee per litre of water, which",
        "is about two tablespoons of ground coffee per cup. Adjust to taste: more",
        "coffee for a stronger cup, less for a milder one.",
    ],
    [
        "THE FRENCH PRESS",
        "Grind the beans coarse, like sea salt. Add the grounds and pour water",
        "just off the boil, around 95 degrees Celsius. Stir once, put the lid on,",
        "and wait four minutes. Then press the plunger down slowly and serve",
        "straight away so it does not turn bitter.",
        "",
        "THE POUR-OVER",
        "Use a medium grind and a paper filter. Pour a little water first to wet",
        "the grounds and let them bloom for 30 seconds, then pour the rest in slow",
        "circles. The whole brew should take about three minutes.",
    ],
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    docs = {
        "remote_work_policy.pdf": REMOTE_WORK_POLICY,
        "solar_system.pdf": SOLAR_SYSTEM,
        "coffee_guide.pdf": COFFEE_GUIDE,
    }
    for name, pages in docs.items():
        path = write_text_pdf(DATA_DIR / name, pages)
        print(f"wrote {path}  ({len(pages)} pages)")
    print(f"\nDone. {len(docs)} PDFs are in {DATA_DIR}.")


if __name__ == "__main__":
    main()
