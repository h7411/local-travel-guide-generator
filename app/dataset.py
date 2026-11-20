"""Dataset of travel interests and local recommendations for fictional cities.

Designed for fine-tuning language models with LoRA to generate personalized travel guides.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TravelRecord:
    id: str
    destination: str
    interests: list[str]
    title: str
    highlights: list[str]
    recommendation: str
    local_tip: str

    def to_prompt(self) -> str:
        interests_str = ", ".join(self.interests)
        return f"Destination: {self.destination} | Interests: {interests_str} -> Guide:"

    def to_completion(self) -> str:
        highlights_str = "; ".join(self.highlights)
        return (
            f" [{self.destination}] {self.title}. Recommended stops: {highlights_str}. "
            f"{self.recommendation} Insider Tip: {self.local_tip}"
        )

    def to_training_pair(self) -> dict[str, str]:
        prompt = self.to_prompt()
        completion = self.to_completion()
        return {
            "prompt": prompt,
            "completion": completion,
            "full_text": f"{prompt}{completion}",
            "destination": self.destination,
            "interests": ", ".join(self.interests),
        }


TRAVEL_DATASET_RAW: list[dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # Solaria: The Sunlit Coastal Citadel
    # -------------------------------------------------------------------------
    {
        "id": "sol-001",
        "destination": "Solaria",
        "interests": ["art", "local coffee"],
        "title": "Sunlit Bohemia: Mosaics and Artisanal Roasts",
        "highlights": ["Sunken Mosaic Gallery", "Café Antiqua in the Old Quarter", "Artisan Glassblowers Lane"],
        "recommendation": "Begin your morning at Café Antiqua savoring a saffron roast brewed over hot sand. Stroll down the Sunken Mosaic Gallery to witness Romanesque floor tiles preserved under clear glass walkways, then visit artisan glassblowing ateliers.",
        "local_tip": "Arrive at Café Antiqua before 9:00 AM to catch the freshly baked fig pastries straight from the stone oven.",
    },
    {
        "id": "sol-002",
        "destination": "Solaria",
        "interests": ["history", "ancient ruins"],
        "title": "Echoes of the Sun King: Fortresses & Catacombs",
        "highlights": ["Citadel of the First Dawn", "Sea Wall Ramparts", "The Sunken Roman Amphitheater"],
        "recommendation": "Ascend the stone battlements of the Citadel of the First Dawn for panoramic views across the turquoise gulf. Walk along the historic 12th-century Sea Wall Ramparts before exploring the submerged archways of the Roman Amphitheater at low tide.",
        "local_tip": "Hire licensed archivist guides at the north citadel gate for access to underground vault inscriptions.",
    },
    {
        "id": "sol-003",
        "destination": "Solaria",
        "interests": ["culinary", "sunset dining"],
        "title": "Coastal Gastronomy & Sunset Terraces",
        "highlights": ["Harbor Fish Guild Hall", "Cliffside Olive Terraces", "Restaurante Marisol"],
        "recommendation": "Spend your late afternoon sampling fresh sea urchin and citrus-cured mackerel at the vibrant Harbor Fish Guild Hall. Reserve an outdoor table at Restaurante Marisol perched upon the western bluffs to dine on seafood paella while the sun sinks into the bay.",
        "local_tip": "Ask for the daily catch special cooked in salt crust paired with regional crisp amber wine.",
    },
    {
        "id": "sol-004",
        "destination": "Solaria",
        "interests": ["nature", "scenic harbors"],
        "title": "Breezes of Lumina Bay & Coastal Cliffs",
        "highlights": ["Lumina Pier Lighthouse", "Smugglers Cove Boardwalk", "Whispering Gull Nature Reserve"],
        "recommendation": "Take a wooden skiff ride along Lumina Pier towards the emerald sea caves of Smugglers Cove. Spend a quiet afternoon birdwatching along the pine-fringed boardwalks of Whispering Gull Nature Reserve.",
        "local_tip": "Tide changes can flood Smugglers Cove beach path; consult harbor-master timetables before setting out.",
    },
    {
        "id": "sol-005",
        "destination": "Solaria",
        "interests": ["nightlife", "live music"],
        "title": "Starlight Melodies & Cellar Acoustics",
        "highlights": ["The Cellar Underground Jazz Vault", "Lantern Piazza", "Ember Rooftop Lounge"],
        "recommendation": "As dusk falls, gather in Lantern Piazza for impromptu acoustic flamenco before stepping down into The Cellar Underground Jazz Vault, housed inside a 400-year-old wine grotto.",
        "local_tip": "The jazz vault does not take phone reservations; queue by 8:30 PM for table seating near the bandstand.",
    },
    {
        "id": "sol-006",
        "destination": "Solaria",
        "interests": ["wellness", "relaxation"],
        "title": "Opal Thermal Springs & Ocean Breeze Spa",
        "highlights": ["The Sunken Thermal Baths", "Aromatherapy Pergola", "Solaria Coastal Thalassotherapy Pool"],
        "recommendation": "Recharge your spirit in the mineral-rich geothermal waters of the Opal Sunken Thermal Baths, naturally heated to 38°C. Follow with a seaweed wrap and lavender meditation session under the shaded Aromatherapy Pergola.",
        "local_tip": "Sunset bath sessions require booking 24 hours in advance and include herbal infusion teas.",
    },

    # -------------------------------------------------------------------------
    # Novalis: The City of Floating Gardens & Canals
    # -------------------------------------------------------------------------
    {
        "id": "nov-001",
        "destination": "Novalis",
        "interests": ["botanical gardens", "canal tours"],
        "title": "Floating Flora & Emerald Waterways",
        "highlights": ["Lotus Barge Gardens", "Grand Grand Canal Gondola Crossing", "The Glasshouse Conservatory"],
        "recommendation": "Drift through the serene waterways aboard an electric wooden gondola weaving between the floating Lotus Barge Gardens. Disembark at the Victorian Glasshouse Conservatory to admire rare bioluminescent orchids.",
        "local_tip": "Early morning canal rides have the calmest water reflections and zero motorized transport ripples.",
    },
    {
        "id": "nov-002",
        "destination": "Novalis",
        "interests": ["night markets", "street food"],
        "title": "Lantern Night Bazaar & Riverside Bites",
        "highlights": ["Canal-side Night Bazaar", "Dumpling Pontoon Alley", "Spiced Cider Floating Dock"],
        "recommendation": "Wander through the shimmering glow of silk lanterns across five linked wooden bridges at the Night Bazaar. Feast on steamed water chestnut buns, spiced broth noodles, and fresh cardamom pastries.",
        "local_tip": "Exchange cash for bazaar copper tokens at entrance kiosks for vendor discounts.",
    },
    {
        "id": "nov-003",
        "destination": "Novalis",
        "interests": ["chamber music", "architecture"],
        "title": "Acoustic Grandeur: Baroque Halls & Bell Towers",
        "highlights": ["The Waterborne Opera House", "St. Jude Floating Campanile", "Archivists Guildhall"],
        "recommendation": "Tour the baroque facades of the Waterborne Opera House, built on oak pilings directly over the river confluence. Attend a twilight string quartet performance where water acoustics provide natural resonance.",
        "local_tip": "Seats in the lower gallery have optimal acoustics due to riverbed sound reflection.",
    },
    {
        "id": "nov-004",
        "destination": "Novalis",
        "interests": ["art", "fine dining"],
        "title": "Refined Palates & Impressionist Ateliers",
        "highlights": ["River Painter Promenade", "Restaurant L'Iris", "Guild of Watercolour Masters"],
        "recommendation": "Watch plein-air painters capture midday light along the Promenade, followed by a curated six-course botanical tasting menu at Restaurant L'Iris featuring edible river flowers.",
        "local_tip": "Reserve the terrace table overlooking the bridge arch for sunset photography.",
    },

    # -------------------------------------------------------------------------
    # Verdantia: The Forest Canopy Metropolis
    # -------------------------------------------------------------------------
    {
        "id": "vrd-001",
        "destination": "Verdantia",
        "interests": ["canopy walks", "eco-trekking"],
        "title": "Skywalks & Ancient Redwood Trails",
        "highlights": ["Cloud-Bridge Suspension Walk", "Giant Redwood Cathedral", "Mist Falls Overlook"],
        "recommendation": "Step high above the misty forest floor across interconnected cedar suspension bridges. Hike the Mossy Ridge Trail leading to the thunderous Mist Falls Overlook where rainbows form in the spray.",
        "local_tip": "Wear waterproof hiking boots with Vibram soles; suspension bridges can be slippery from canopy mist.",
    },
    {
        "id": "vrd-002",
        "destination": "Verdantia",
        "interests": ["tea houses", "relaxation"],
        "title": "Canopy Teas & Forest Whispers",
        "highlights": ["The Cedar Branch Tea Pavilion", "Fern Valley Meditation Rocks", "Wild Honey Farmhouse"],
        "recommendation": "Sip small-batch mountain oolong steeped in volcanic spring water at the open-air Cedar Branch Tea Pavilion. Unwind listening to cedar-waxwings amidst the ferns.",
        "local_tip": "Try the smoked pine needle tea served with fresh honeycomb biscuits.",
    },
    {
        "id": "vrd-003",
        "destination": "Verdantia",
        "interests": ["wildlife", "photography"],
        "title": "Canopy Fauna & Sunrise Photography",
        "highlights": ["Emerald Flycatcher Blind", "High-Bough Observation Tower", "Glow-Beetle Hollow"],
        "recommendation": "Ascend to the High-Bough Observation Tower before dawn to capture sunrise lighting the misty treetops. Spot rare tree frogs, flying squirrels, and iridescent canopy kingfishers.",
        "local_tip": "A 70-300mm telephoto lens and monopod are recommended for stabilization on wooden blinds.",
    },

    # -------------------------------------------------------------------------
    # Aethelgard: The Mountain Keep & Starlit Peaks
    # -------------------------------------------------------------------------
    {
        "id": "aeth-001",
        "destination": "Aethelgard",
        "interests": ["alpine climbing", "stargazing"],
        "title": "High Glaciers & Celestial Observatories",
        "highlights": ["Eagle Peak Ridge Via Ferrata", "The High Peak Dome Observatory", "Glacial Melt Basin"],
        "recommendation": "Traverse the thrilling steel-cabled Via Ferrata along Eagle Peak Ridge. At nightfall, view deep sky nebulae through the research-grade 30-inch telescope at High Peak Dome Observatory.",
        "local_tip": "Even in mid-summer, night temperatures drop below freezing; bring thermal base layers and windbreakers.",
    },
    {
        "id": "aeth-002",
        "destination": "Aethelgard",
        "interests": ["thermal baths", "hearth taverns"],
        "title": "Volcanic Soaks & Mountain Roasts",
        "highlights": ["Frost-Rim Thermal Caldera", "The Iron Kettle Hearth Tavern", "Alpine Pine Sauna"],
        "recommendation": "Soak in steaming thermal mineral pools surrounded by snowcapped fir trees. Afterwards, gather around the communal stone hearth at The Iron Kettle Tavern for hearty venison stew and crusty sourdough rye.",
        "local_tip": "Order the spiced elderberry mulled cider to warm up after your thermal soak.",
    },
    {
        "id": "aeth-003",
        "destination": "Aethelgard",
        "interests": ["folklore", "artisan crafts"],
        "title": "Carvers Guild & Mountain Legends",
        "highlights": ["Blacksmith & Woodcarver Quadrangle", "The Stone Chronicler Museum", "Runestone Path"],
        "recommendation": "Explore stone-built workshops where master smiths hand-forge Damascus blades and woodcarvers fashion intricate spruce masks. Follow the historic Runestone Path recounting ancient mountaineer legends.",
        "local_tip": "The woodcarvers hold live chisel demonstrations every day at 11:00 AM.",
    },
]


def load_dataset() -> list[TravelRecord]:
    """Load and return all travel records."""
    return [TravelRecord(**item) for item in TRAVEL_DATASET_RAW]


def get_dataset_stats() -> dict[str, Any]:
    """Return summary statistics of the travel dataset."""
    records = load_dataset()
    destinations = sorted(list({r.destination for r in records}))
    all_interests: set[str] = set()
    for r in records:
        all_interests.update(r.interests)

    return {
        "total_records": len(records),
        "destinations": destinations,
        "destinations_count": len(destinations),
        "unique_interests": sorted(list(all_interests)),
        "unique_interests_count": len(all_interests),
    }


def get_training_pairs() -> list[dict[str, str]]:
    """Return all formatted input-output prompt-completion pairs."""
    records = load_dataset()
    return [r.to_training_pair() for r in records]
