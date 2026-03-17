#pragma once

#include "darknet_internal.hpp"

/// Key-value pair for configuration options.
struct kvp
{
	std::string key;
	std::string val;
	int used = 0;
};

/// Read the .data file.
list *read_data_cfg(const char *filename);
list *read_data_cfg(const std::string & filename);

/** Parse a key-value pair from a single line of text that came from a @p .cfg or @p .data file.
 *
 * @returns @p false if the line does not contain a key-value pair.
 * @returns @p true if a key-value pair was parsed and stored in @p options.
 */
bool read_option(const std::string & s, list *options);

/// @deprecated Use read_option(const std::string &, list*) instead.
[[deprecated("Use read_option(const std::string &, list*) instead")]]
int read_option(char *s, list *options);

void option_insert(list *l, const std::string & key, const std::string & val);
std::string option_find(list *l, const std::string & key);
std::string option_find_str(list *l, const std::string & key, const std::string & def);
std::string option_find_str_quiet(list *l, const std::string & key, const std::string & def);
int option_find_int(list *l, const std::string & key, int def);
int option_find_int_quiet(list *l, const std::string & key, int def);
float option_find_float(list *l, const std::string & key, float def);
float option_find_float_quiet(list *l, const std::string & key, float def);
void option_unused(list *l);
