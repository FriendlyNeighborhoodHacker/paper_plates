"Paper Plates Image Generation" Python Program

I have a google spreadsheet with three columns:
- name
- superlative
- drawing_instructions

I want to generate "paper plate superlative images" for an end-of-year 11th grade presentation where we will go through and give everyone in the class superlative awards.

I want to write a python program to fetch the data from Google Sheets, and one by one send them to the latest OpenAI image generation api to generate the image, and store the image in a local directory with a filename based on the name of the person.

I have an OPENAI_API_KEY - I am subscribed to the $100 / month plan that allows image generation.

I have developed a "prompt intro" in the file in this directory "prompt_intro.txt".  I'd like that intro to be sent as the start of the image generation prompt.

The program should have the following configuration (in a configuration file that should NOT be seen by git, so I also need a gitignore).  It can be a YAML file.
- OPENAI_API_KEY
- GOOGLE_SHEETS_URL

It should go to the Google Sheets URL, download the contents as a CSV (it will have three columns), extract an array of "name", "superlative", and "drawing_instructions".

Then, it should process each row by sending both the superlative and drawing_instructions (not the name) to OpenAI using the most advanced image generation model (gpt-image-1), and save the image as a filename in a directory "generated_images".

Already-generated images are skipped on re-runs (delete the file to regenerate).

There is also a test script `test_sheets.py` that can be run independently to verify the Google Sheets fetch without making any OpenAI API calls.
