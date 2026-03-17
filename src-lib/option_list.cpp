#include "option_list.hpp"
#include "darknet_internal.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();
}


list *read_data_cfg(const std::string & filename)
{
	TAT(TATPARMS);

	FILE * file = fopen(filename.c_str(), "r");
	if (file == nullptr)
	{
		file_error(filename.c_str(), DARKNET_LOC);
	}

	int line_number = 0;
	list *options = make_list();
	std::string line;
	while (!(line = fgetl(file)).empty() || !feof(file))
	{
		++line_number;
		line = Darknet::trim(line);
		if (line.empty() || line[0] == '#' || line[0] == ';')
		{
			continue;
		}
		if (!read_option(line, options))
		{
			Darknet::display_warning_msg("WARNING: failed to parse line #" + std::to_string(line_number) + " in " + filename + ": " + line + "\n");
		}
	}

	fclose(file);

	/* There is a limited number of options that typically exists in the .data files.  We should expect the following:
	 *
	 *		classes = <number>
	 *		train = <filename>
	 *		valid = <filename>
	 *		names = <filename>
	 *		backup = <directory>
	 */

	std::string str = option_find(options, "classes");
	if (str.empty())
	{
		Darknet::display_warning_msg("WARNING: expected to find \"classes=...\" in " + filename + "\n");
	}
	else
	{
		const int classes = std::stoi(str);
		if (classes <= 0 or classes >= 50)
		{
			Darknet::display_warning_msg("WARNING: unusual number of classes (" + std::to_string(classes) + ") in " + filename + "\n");
		}
	}

	for (const std::string fn : {"train", "valid", "names"})
	{
		str = option_find(options, fn);
		if (str.empty())
		{
			Darknet::display_warning_msg("WARNING: expected to find \"" + fn + "=...\" in " + filename + "\n");
		}
		else
		{
			// does this file actually exist?
			if (std::filesystem::exists(str) == false)
			{
				Darknet::display_warning_msg("WARNING: file " + str + " does not seem to exist (\"" + fn + "=...\") in " + filename + "\n");
			}
		}
	}

	str = option_find(options, "backup");
	if (str.empty())
	{
		Darknet::display_warning_msg("WARNING: expected to find \"backup=...\" in " + filename + "\n");
	}
	else
	{
		if (std::filesystem::is_directory(str) == false)
		{
			Darknet::display_warning_msg("WARNING: \"" + str + "\" does not seem to be a valid directory for \"backup=...\" in " + filename + "\n");
		}
	}

	// see if there are options we don't recognize
	node * n = options->front;
	while (n)
	{
		kvp * p = (kvp *)n->val;
		if (p->used == 0)
		{
			Darknet::display_warning_msg("WARNING: unexpected option \"" + p->key + "=" + p->val + "\" in " + filename + "\n");
		}
		n = n->next;
	}

	return options;
}


list *read_data_cfg(const char *filename)
{
	TAT(TATPARMS);
	return read_data_cfg(std::string(filename ? filename : ""));
}


bool read_option(const std::string & s, list *options)
{
	TAT(TATPARMS);

	// expect the line to be KEY=VAL so look for the "="
	size_t pos = s.find('=');
	if (pos == std::string::npos)
	{
		return false;
	}

	std::string key = Darknet::trim(s.substr(0, pos));
	std::string val = Darknet::trim(s.substr(pos + 1));

	option_insert(options, key, val);
	return true;
}


int read_option(char *s, list *options)
{
	TAT(TATPARMS);
	// Deprecated wrapper
	return read_option(std::string(s ? s : ""), options) ? 1 : 0;
}


void option_insert(list *l, const std::string & key, const std::string & val)
{
	TAT(TATPARMS);

	kvp* p = new kvp{key, val, 0};
	list_insert(l, p);
}


void option_unused(list *l)
{
	TAT(TATPARMS);

	kvp * previous_kvp = nullptr;

	node *n = l->front;
	while(n)
	{
		kvp *p = (kvp *)n->val;
		if (!p->used)
		{
			if (previous_kvp)
			{
				// attempt to give some context as to where the error is happening in the .cfg file
				*cfg_and_state.output													<< std::endl
					<< "Last option was: " << previous_kvp->key	<< "=" << previous_kvp->val << std::endl
					<< "Unused option is " << p->key			<< "=" << p->val			<< std::endl;
			}
			darknet_fatal_error(DARKNET_LOC, "invalid, unused, or unrecognized option: %s=%s", p->key.c_str(), p->val.c_str());
		}
		previous_kvp = p;
		n = n->next;
	}
}


std::string option_find(list *l, const std::string & key)
{
	TAT(TATPARMS);

	node *n = l->front;
	while(n)
	{
		kvp *p = (kvp *)n->val;
		if (p->key == key)
		{
			p->used = 1;
			return p->val;
		}
		n = n->next;
	}
	return {};
}


std::string option_find_str(list *l, const std::string & key, const std::string & def)
{
	TAT(TATPARMS);

	std::string v = option_find(l, key);
	if (!v.empty())
	{
		return v;
	}

	if (!def.empty())
	{
		*cfg_and_state.output << key << ": Using default \"" << def << "\"" << std::endl;
	}

	return def;
}


std::string option_find_str_quiet(list *l, const std::string & key, const std::string & def)
{
	TAT(TATPARMS);

	std::string v = option_find(l, key);
	if (!v.empty())
	{
		return v;
	}
	return def;
}


int option_find_int(list *l, const std::string & key, int def)
{
	TAT(TATPARMS);

	std::string v = option_find(l, key);
	if (!v.empty())
	{
		return std::stoi(v);
	}

	*cfg_and_state.output << key << ": Using default \"" << def << "\"" << std::endl;

	return def;
}


int option_find_int_quiet(list *l, const std::string & key, int def)
{
	TAT(TATPARMS);

	std::string v = option_find(l, key);
	if (!v.empty())
	{
		return std::stoi(v);
	}

	return def;
}


float option_find_float_quiet(list *l, const std::string & key, float def)
{
	TAT(TATPARMS);

	std::string v = option_find(l, key);
	if (!v.empty())
	{
		return std::stof(v);
	}

	return def;
}


float option_find_float(list *l, const std::string & key, float def)
{
	TAT(TATPARMS);

	std::string v = option_find(l, key);
	if (!v.empty())
	{
		return std::stof(v);
	}

	*cfg_and_state.output << key << ": Using default \"" << def << "\"" << std::endl;

	return def;
}
