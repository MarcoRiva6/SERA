from smolagents import OpenAIServerModel, CodeAgent

model = OpenAIServerModel(
    model_id="local-model",
    api_base="http://localhost:1234/v1",
    api_key="not-needed",
)

# Create an agent with the default toolbox
agent = CodeAgent(
    name="AssistantWithDefaultTools",
    model=model,
    tools=[],  # Your custom tools would go here
    add_base_tools=True  # This adds DuckDuckGo search, Python interpreter, and Transcriber
)

text = """

You are a movie recommendation system. Analyze the provided movie dataset and respond to the user's request.

Dataset:
[
  {
    "IMDB_id":"tt0457655",
    "title":"After the Wedding",
    "year":2006,
    "genre":"Drama",
    "directors":"Susanne Bier",
    "writers":"Susanne Bier|Anders Thomas Jensen",
    "main_cast":"Mads Mikkelsen|Sidse Babett Knudsen|Rolf Lassg\u00e5rd",
    "duration_min":120,
    "AVG_score":7.6,
    "number_of_votes":38694
  },
  {
    "IMDB_id":"tt0816692",
    "title":"Interstellar",
    "year":2014,
    "genre":"Adventure|Drama|Sci-Fi",
    "directors":"Christopher Nolan",
    "writers":"Jonathan Nolan|Christopher Nolan",
    "main_cast":"Matthew McConaughey|Anne Hathaway|Ellen Burstyn",
    "duration_min":169,
    "AVG_score":8.7,
    "number_of_votes":2408967
  },
  {
    "IMDB_id":"tt0964185",
    "title":"Tetro",
    "year":2009,
    "genre":"Drama",
    "directors":"Francis Ford Coppola",
    "writers":"Francis Ford Coppola",
    "main_cast":"Vincent Gallo|Maribel Verd\u00fa|Silvia P\u00e9rez|Rodrigo de la Serna",
    "duration_min":127,
    "AVG_score":6.8,
    "number_of_votes":14130
  },
  {
    "IMDB_id":"tt1020558",
    "title":"Centurion",
    "year":2010,
    "genre":"Action|Drama|History",
    "directors":"Neil Marshall",
    "writers":"Neil Marshall",
    "main_cast":"Dominic West|Andreas Wisniewski|Dave Legeno",
    "duration_min":97,
    "AVG_score":6.3,
    "number_of_votes":88690
  },
  {
    "IMDB_id":"tt0104231",
    "title":"Far and Away",
    "year":1992,
    "genre":"Adventure|Drama|Romance",
    "directors":"Ron Howard",
    "writers":"Bob Dolman|Ron Howard",
    "main_cast":"Tom Cruise|Nicole Kidman|Thomas Gibson|Robert Prosky|Barbara Babcock",
    "duration_min":140,
    "AVG_score":6.6,
    "number_of_votes":71298
  },
  {
    "IMDB_id":"tt0120036",
    "title":"Rosewood",
    "year":1997,
    "genre":"Action|Drama|History",
    "directors":"John Singleton",
    "writers":"Gregory Poirier",
    "main_cast":"Jon Voight|Ving Rhames|Don Cheadle|Bruce McGill|Loren Dean",
    "duration_min":140,
    "AVG_score":7.2,
    "number_of_votes":9220
  },
  {
    "IMDB_id":"tt0413893",
    "title":"The Taste of Tea",
    "year":2004,
    "genre":"Comedy|Fantasy",
    "directors":"Katsuhito Ishii",
    "writers":"Katsuhito Ishii",
    "main_cast":"Tadanobu Asano|Tatsuya Gash\u00fbin",
    "duration_min":143,
    "AVG_score":7.6,
    "number_of_votes":7696
  },
  {
    "IMDB_id":"tt0414055",
    "title":"Elizabeth: The Golden Age",
    "year":2007,
    "genre":"Biography|Drama|History",
    "directors":"Shekhar Kapur",
    "writers":"William Nicholson|Michael Hirst",
    "main_cast":"Cate Blanchett|Clive Owen|Geoffrey Rush|Jordi Moll\u00e0",
    "duration_min":114,
    "AVG_score":6.8,
    "number_of_votes":77355
  },
  {
    "IMDB_id":"tt0119815",
    "title":"Four Days in September",
    "year":1997,
    "genre":"Action|Drama|History",
    "directors":"Bruno Barreto",
    "writers":"Fernando Gabeira|Leopoldo Serran",
    "main_cast":"Alan Arkin|Pedro Cardoso|Pedro Cardoso|Fernanda Torres|Luiz Fernando Guimar\u00e3es",
    "duration_min":110,
    "AVG_score":7.4,
    "number_of_votes":6156
  },
  {
    "IMDB_id":"tt0455590",
    "title":"The Last King of Scotland",
    "year":2006,
    "genre":"Biography|Drama|History",
    "directors":"Kevin Macdonald",
    "writers":"Peter Morgan|Jeremy Brock",
    "main_cast":"James McAvoy|Forest Whitaker|Gillian Anderson|Kerry Washington|Simon McBurney",
    "duration_min":123,
    "AVG_score":7.6,
    "number_of_votes":202873
  },
  {
    "IMDB_id":"tt0918940",
    "title":"The Legend of Tarzan",
    "year":2016,
    "genre":"Action|Adventure|Drama",
    "directors":"David Yates",
    "writers":"Craig Brewer|Edgar Rice Burroughs",
    "main_cast":"Alexander Skarsg\u00e5rd|Alexander Skarsg\u00e5rd|Christoph Waltz",
    "duration_min":110,
    "AVG_score":6.2,
    "number_of_votes":195027
  },
  {
    "IMDB_id":"tt0113670",
    "title":"A Little Princess",
    "year":1995,
    "genre":"Drama|Family|Fantasy",
    "directors":"Alfonso Cuar\u00f3n",
    "writers":"Frances Hodgson Burnett|Richard LaGravenese|Elizabeth Chandler",
    "main_cast":"Liesel Matthews|Eleanor Bron|Liam Cunningham|Liam Cunningham|Rusty Schwimmer",
    "duration_min":97,
    "AVG_score":7.6,
    "number_of_votes":38402
  },
  {
    "IMDB_id":"tt0216216",
    "title":"The 6th Day",
    "year":2000,
    "genre":"Action|Mystery|Sci-Fi",
    "directors":"Roger Spottiswoode",
    "writers":"Cormac Wibberley|Marianne Wibberley",
    "main_cast":"Arnold Schwarzenegger|Michael Rapaport|Tony Goldwyn|Michael Rooker|Sarah Wynter",
    "duration_min":123,
    "AVG_score":5.9,
    "number_of_votes":131963
  },
  {
    "IMDB_id":"tt0428870",
    "title":"A Moment to Remember",
    "year":2004,
    "genre":"Drama|Romance",
    "directors":"John H. Lee",
    "writers":"John H. Lee",
    "main_cast":"Jung Woo-sung|Baek Jong-hak",
    "duration_min":117,
    "AVG_score":8.1,
    "number_of_votes":26864
  },
  {
    "IMDB_id":"tt1099212",
    "title":"Twilight",
    "year":2008,
    "genre":"Drama|Fantasy|Romance",
    "directors":"Catherine Hardwicke",
    "writers":"Melissa Rosenberg",
    "main_cast":"Kristen Stewart|Billy Burke",
    "duration_min":122,
    "AVG_score":5.3,
    "number_of_votes":513242
  },
  {
    "IMDB_id":"tt0960790",
    "title":"Rabbit Without Ears",
    "year":2007,
    "genre":"Comedy|Romance",
    "directors":"Til Schweiger",
    "writers":"Til Schweiger",
    "main_cast":"Til Schweiger|Matthias Schweigh\u00f6fer|J\u00fcrgen Vogel",
    "duration_min":116,
    "AVG_score":6.5,
    "number_of_votes":19144
  },
  {
    "IMDB_id":"tt0301414",
    "title":"Man on the Train",
    "year":2002,
    "genre":"Crime|Drama|Thriller",
    "directors":"Patrice Leconte",
    "writers":"Patrick Cauvin",
    "main_cast":"Jean Rochefort|Johnny Hallyday|Jean-Fran\u00e7ois St\u00e9venin|Charlie Nelson|Pascal Parmentier",
    "duration_min":90,
    "AVG_score":7.1,
    "number_of_votes":7737
  },
  {
    "IMDB_id":"tt0335119",
    "title":"Girl with a Pearl Earring",
    "year":2003,
    "genre":"Biography|Drama|Romance",
    "directors":"Peter Webber",
    "writers":"Olivia Hetreed",
    "main_cast":"Scarlett Johansson|Colin Firth|Tom Wilkinson|Judy Parfitt|Cillian Murphy",
    "duration_min":100,
    "AVG_score":6.9,
    "number_of_votes":84777
  },
  {
    "IMDB_id":"tt0120885",
    "title":"Wag the Dog",
    "year":1997,
    "genre":"Comedy|Drama",
    "directors":"Barry Levinson",
    "writers":"Larry Beinhart|Hilary Henkin|David Mamet",
    "main_cast":"Dustin Hoffman|Robert De Niro|Anne Heche|Woody Harrelson|Denis Leary",
    "duration_min":97,
    "AVG_score":7.1,
    "number_of_votes":91730
  },
  {
    "IMDB_id":"tt0102587",
    "title":"Only Yesterday",
    "year":1991,
    "genre":"Animation|Drama|Romance",
    "directors":"Isao Takahata",
    "writers":"Hotaru Okamoto|Isao Takahata",
    "main_cast":"Miki Imai|Toshir\u00f4 Yanagiba|Yoko Honna|Michie Terada|Masahiro Ito",
    "duration_min":119,
    "AVG_score":7.6,
    "number_of_votes":40904
  },
  {
    "IMDB_id":"tt0337876",
    "title":"Birth",
    "year":2004,
    "genre":"Drama|Fantasy|Mystery",
    "directors":"Jonathan Glazer",
    "writers":"Jean-Claude Carri\u00e8re|Milo Addica|Jonathan Glazer",
    "main_cast":"Nicole Kidman|Lauren Bacall|Danny Huston|Alison Elliott",
    "duration_min":100,
    "AVG_score":6.3,
    "number_of_votes":43513
  },
  {
    "IMDB_id":"tt0456396",
    "title":"The Child",
    "year":2005,
    "genre":"Crime|Drama|Romance",
    "directors":"Jean-Pierre Dardenne|Luc Dardenne",
    "writers":"Jean-Pierre Dardenne|Luc Dardenne",
    "main_cast":"J\u00e9r\u00e9mie Renier|Fabrizio Rongione|Olivier Gourmet",
    "duration_min":95,
    "AVG_score":7.4,
    "number_of_votes":19933
  },
  {
    "IMDB_id":"tt0168629",
    "title":"Dancer in the Dark",
    "year":2000,
    "genre":"Crime|Drama|Musical",
    "directors":"Lars von Trier",
    "writers":"Lars von Trier|Sj\u00f3n",
    "main_cast":"Bj\u00f6rk|Catherine Deneuve|David Morse|Peter Stormare|Joel Grey",
    "duration_min":135,
    "AVG_score":7.9,
    "number_of_votes":121892
  },
  {
    "IMDB_id":"tt1196956",
    "title":"One Chance",
    "year":2013,
    "genre":"Biography|Comedy|Drama",
    "directors":"David Frankel",
    "writers":"Justin Zackham",
    "main_cast":"James Corden|Julie Walters|Colm Meaney|Mackenzie Crook",
    "duration_min":103,
    "AVG_score":6.8,
    "number_of_votes":13829
  },
  {
    "IMDB_id":"tt11327514",
    "title":"Uncle Frank",
    "year":2020,
    "genre":"Comedy|Drama",
    "directors":"Alan Ball",
    "writers":"Alan Ball",
    "main_cast":"Paul Bettany|Peter Macdissi|Steve Zahn|Judy Greer",
    "duration_min":95,
    "AVG_score":7.3,
    "number_of_votes":24312
  },
  {
    "IMDB_id":"tt0116483",
    "title":"Happy Gilmore",
    "year":1996,
    "genre":"Comedy|Sport",
    "directors":"Dennis Dugan",
    "writers":"Tim Herlihy|Adam Sandler",
    "main_cast":"Adam Sandler|Christopher McDonald|Julie Bowen|Frances Bay|Carl Weathers",
    "duration_min":92,
    "AVG_score":7.0,
    "number_of_votes":295888
  },
  {
    "IMDB_id":"tt0373861",
    "title":"The Story of the Weeping Camel",
    "year":2003,
    "genre":"Documentary|Drama|Family",
    "directors":"Luigi Falorni",
    "writers":"Luigi Falorni",
    "main_cast":null,
    "duration_min":93,
    "AVG_score":7.4,
    "number_of_votes":6517
  },
  {
    "IMDB_id":"tt0456912",
    "title":"A Bittersweet Life",
    "year":2005,
    "genre":"Action|Crime|Drama",
    "directors":"Kim Jee-woon",
    "writers":"Kim Jee-woon",
    "main_cast":"Lee Byung-hun",
    "duration_min":119,
    "AVG_score":7.5,
    "number_of_votes":45867
  },
  {
    "IMDB_id":"tt1038686",
    "title":"Legion",
    "year":2010,
    "genre":"Action|Fantasy|Horror",
    "directors":"Scott Stewart",
    "writers":"Peter Schink|Scott Stewart",
    "main_cast":"Paul Bettany|Dennis Quaid|Charles S. Dutton|Lucas Black|Tyrese Gibson",
    "duration_min":100,
    "AVG_score":5.3,
    "number_of_votes":117619
  },
  {
    "IMDB_id":"tt0257756",
    "title":"High Crimes",
    "year":2002,
    "genre":"Crime|Drama|Mystery",
    "directors":"Carl Franklin",
    "writers":"Joseph Finder|Yuri Zeltser|Grace Cary Bickley",
    "main_cast":"Jim Caviezel|Jim Caviezel|Morgan Freeman|Ashley Judd|Adam Scott",
    "duration_min":115,
    "AVG_score":6.4,
    "number_of_votes":50680
  }
]

User Request:
Detesto film come 'The 6th Day', suggerisci qualcosa di completamente diverso

Instructions:
- Analyze the movies in the dataset
- For similarity requests, find movies with similar genres, themes, or characteristics
- For negative similarity requests ("opposto", "opposite", "contrario"), find movies that are opposite in style, genre, or theme
- Return the results as a list of movie IDs (tt codes) that best match the request
- Consider factors like genre, year, rating, plot, and other available attributes when available
- Provide exactly 10 recommendations when possible
- If some data is missing (NULL values), use the available information to make the best recommendations

Response Format:
Return only the movie IDs (tt codes) separated by |, for example:
tt0120338|tt0167260|tt0290334|tt0372784|tt0449088|tt0475783|tt0800080|tt0848228|tt1254207|tt1300854
"""

response = agent.run(text)
print(f"Agent response: {response}")