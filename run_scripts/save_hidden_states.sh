# Activate virtual environment
source activate llmtt

for relation in applies_to_jurisdiction capital capital_of continent \
                country_of_citizenship developer field_of_work \
                headquarters_location instrument language_of_work_or_name \
                languages_spoken location_of_formation manufacturer \
                native_language occupation official_language \
                owned_by place_of_birth place_of_death religion
do
    for lang in en ca es fr hu ja ko nl ru uk vi zh
    do
        python llm-transparency-tool-bak/llm_transparency_tool/server/app-backend.py \
            --config_file llm-transparency-tool-bak/config/backend_llama-2-7b.json \
            --dataset_path klar/${lang} \
            --few_shot_demo 3 \
            --relation $relation \
            --language $lang \
            --output_path llm-transparency-tool-bak/YOUR_RESULTS_DIR \
            --log_hidden_states True
    done
done